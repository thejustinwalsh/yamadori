"""Deploy LAYOUT V2 (operator, 2026-09-29) with the line bench/kv_rank.py measured (`--layout v2`, its
default). In the style of bench/deploy_kv_rank.py: every file it touches is backed up and put back if a check
fails before the restart; then the old start-stack.bat launchers are killed, ALL services restart through the
Scheduled Task, this waits for health, the watchdog's task is restarted (a running watchdog keeps the script
it started with), and the served /props is re-recorded into the offline fixture. scripts/deploy_check.py is
run separately: a deploy is not good until it exits 0.

THE OPERATOR'S DECISIONS (2026-09-29)

  (a) "The point is to get more context at speed in vram, so decider was the only thing that needed room."
  (b) "Vision can go to second card and swap in and out." (reverses 2026-09-27's fold of vision into the
      main model on the 5060 Ti)
  (The single-model decision -- the second brain removed -- is a separate removal, not this deploy.)

WHAT CHANGES

  config.yaml, `bonsai`   --mmproj REMOVED (the projector cost ~24.8k cells of the VRAM line: the 2026-09-27
                          fit, docs/ENGINES.md "Deployed"); --kv-vram-cells N; -c 2N + LANE (the lane:
                          budget.LANE_TOKENS). Nothing else in its argv or env changes (checked).
  config.yaml, `bonsai-vision`
                          the retired A4000 entry RESTORED (config.yaml.bak-20260927-233721, the last config
                          that had it), on the ORIGINAL trunk the main model serves (operator 2026-09-27),
                          with the projector; ttl 300; and its `ondemand` group (swap, exclusive), as before
                          the fold. mcp/gpu_room.py decides what leaves the A4000 for it (SIZES row
                          `bonsai-vision`).
  scripts/start-stack.bat, scripts/watchdog.ps1
                          the 2026-09-28 kv-layout block (YAMADORI_MAIN_CAP=N_old, YAMADORI_CHILD_TOKENS)
                          is REPLACED by YAMADORI_MAIN_CAP = N - LANE (the lane's cells come out of the line:
                          budget.main_cap) and YAMADORI_LANE_TOKENS = LANE, for the proxy, the tools API and
                          the worker (a watchdog restart inherits the watchdog's environment: `Env2`).
  mcp/fixtures/bonsai_props.json
                          re-recorded from the SERVED /props after the restart (n_ctx, total_slots,
                          kv_vram_cells, modalities.vision) and checked against what was intended. n_ctx is
                          the SLOT's window, which llama-server caps at the model's trained 262,144
                          (tools/server/server-context.cpp:1203-1206 in the 0041 build: "the slot context
                          ... exceeds the training context of the model - capping"), so the intended n_ctx
                          is min(2N + LANE, 262,144), not -c.

    python bench/deploy_layout_v2.py --line N --preview   # every edit as a full diff; nothing is written
    python bench/deploy_layout_v2.py --line N --dry       # make and check the edits, then put them back
    python bench/deploy_layout_v2.py --line N             # deploy: edit, check, restart, wait, re-record

The edits are pure functions of the files' text (edit_config, edit_bat, edit_watchdog), so --preview shows
them without touching the live files: llama-swap reloads a changed config.yaml, which a --dry run does twice.
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "mcp"))
import budget                                                        # noqa: E402
import deploy_kv_rank as dkr                                         # noqa: E402

FILES = ["config.yaml", "scripts/start-stack.bat", "scripts/watchdog.ps1", "mcp/fixtures/bonsai_props.json"]
MARK = "layout v2 (bench/deploy_layout_v2.py)"
OLD_MARKS = (dkr.MARK,)            # the 2026-09-28 kv layout's env block, replaced
LANE = budget.LANE_TOKENS
TRAINED_WINDOW = 262144            # the model's n_ctx_train (docs/ENGINES.md "The target")
A4000_UUID = "GPU-43e37d0c-4104-9056-2552-6109d4d3382c"
VISION_ID = "bonsai-vision"
VISION_TRUNK = "Ternary-Bonsai-2-27B-PTQ1_0.gguf"
MMPROJ = "Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf"
WATCHDOG_TASK = "llama-stack-watchdog"
STACK_TASK = "llama-stack"

read, write, _lf = dkr.read, dkr.write, dkr._lf

VISION_BANNER_OLD = """  # =====================================================================
  # VISION variant, ON DEMAND (ttl 900). Same weights plus the multimodal
  # projector, so the stack can actually look at a flamegraph, a RenderDoc
  # capture, a misrendered frame or a whiteboard photo.
  # Separate process because mmproj would cost the primary ~0.6 GiB of
  # cache, and context matters more there than images do.
  # =====================================================================
"""

VISION_ENTRY = f"""  # =====================================================================
  # VISION, ON DEMAND on the A4000 (ttl 300). The main model's trunk plus the
  # multimodal projector, so the stack can look at a flamegraph, a RenderDoc
  # capture, a misrendered frame or a whiteboard photo.
  # {MARK}, operator 2026-09-29: "Vision can go to second card and swap in
  # and out." -- the 2026-09-27 fold of the projector into `bonsai` cost ~24.8k
  # cells of the 5060 Ti's VRAM line (docs/ENGINES.md "Deployed"), and "the
  # point is to get more context at speed in vram". The entry is the one
  # retired on 2026-09-27 (config.yaml.bak-20260927-233721), on the ORIGINAL
  # trunk the main model serves (operator 2026-09-27: back to the original
  # trunk). mcp/vision.py sends attached images and yama_describe_image here
  # whenever the main model has no projector (its /props modalities.vision);
  # mcp/gpu_room.py makes room on the A4000 first (SIZES `bonsai-vision`).
  # =====================================================================
  "{VISION_ID}":
    name: "Bonsai 2 27B ternary + vision (A4000)"
    env:
      - "CUDA_VISIBLE_DEVICES={A4000_UUID}"
    cmd: |
      ${{server}}
      --port ${{PORT}}
      -m ${{models}}/{VISION_TRUNK}
      --mmproj ${{models}}/{MMPROJ}
      -dev CUDA0
      -ngl 999
      -c 16384
      --cache-type-k q8_0 --cache-type-v q8_0
      -fa on
      -b 1024 -ub 512
      --jinja
      ${{sampling}}
    # 300 s (operator, 2026-09-23; was 900). What else leaves the A4000 when
    # vision loads, and when vision itself leaves for a draw or a search, is
    # decided by mcp/gpu_room.py; the ttl only frees the card when nothing
    # asks for it.
    ttl: 300

"""

ONDEMAND_GROUP = f"""        # Vision on demand ({MARK}; restored as it was before the 2026-09-27
        # fold). It stays exclusive because the numbers say it cannot share with
        # search anyway: its own llama-server with its own copy of the 27B,
        # estimated 8,265-9,449 MiB (weights 5.95 GiB + mmproj 0.59 + KV@16384 +
        # a 1-2 GiB compute buffer; never measured alone), against 8,605 MiB
        # free beside retrieval + Laya (7,565 MiB measured). gpu_room decides
        # first; this is the backstop.
        "ondemand":
          swap: true
          exclusive: true
          members:
            - "{VISION_ID}"

"""


def _bonsai_block(c: str) -> tuple[int, int]:
    i0 = c.index('\n  "bonsai":\n')
    i1 = re.search(r'\n  "[^"]+":\n', c[i0 + 5:]).start() + i0 + 5
    return i0, i1


def edit_config(text: str, n: int, lane: int = LANE) -> str:
    """config.yaml with layout v2. Idempotent: a second run changes nothing."""
    c, nl = _lf(text)
    ctx = 2 * n + lane
    i0, i1 = _bonsai_block(c)
    blk = c[i0:i1]
    if blk.count("\n      -c ") != 1:
        raise ValueError("bonsai's -c line not found once")
    blk = re.sub(r"\n      -c \d+\n", f"\n      -c {ctx}\n", blk)
    kv = re.findall(r"\n      --kv-vram-cells \d+\n", blk)
    if len(kv) != 1:
        raise ValueError(f"bonsai: --kv-vram-cells found {len(kv)} times (the v1 layout has it once)")
    blk = blk.replace(kv[0], f"\n      --kv-vram-cells {n}\n")
    mm = re.findall(r"\n      --mmproj [^\n]+\n", blk)
    if len(mm) > 1:
        raise ValueError("bonsai: more than one --mmproj")
    if mm:
        blk = blk.replace(mm[0], (
            f"\n      # {MARK}, operator 2026-09-29: --mmproj REMOVED (\"Vision can go to second card and\n"
            "      # swap in and out\"): the projector cost ~24.8k cells of this card's VRAM line (the\n"
            "      # 2026-09-27 fit). Vision is `bonsai-vision` on the A4000 (below).\n"), 1)
    note = (f"      # {MARK}, operator 2026-09-29 (\"the point is to get more context at speed in vram, so\n"
            f"      # decider was the only thing that needed room\"): 3 slots = two conversation slots + THE LANE\n"
            f"      # (the decider and small side calls; budget.LANE_TOKENS = {lane}, kept in VRAM at\n"
            f"      # kv_rank slots.RANK_LANE); the VRAM line N = {n} measured by bench/kv_rank.py (v2);\n"
            f"      # -c = 2N + the lane; the main cap N - the lane (budget.main_cap); the MTP draft context\n"
            "      # sized for a 16384-token window per slot (0036)\n")
    old_v1 = re.search(r"      # 2026-09-28 \(operator; " + re.escape(dkr.MARK) + r"\):[^\n]*\n(?:      #[^\n]*\n){2}",
                       blk)
    old_v2 = re.search(r"      # " + re.escape(MARK) + r", operator 2026-09-29 \(\"the point[^\n]*\n"
                       r"(?:      #[^\n]*\n){5}", blk)
    if old_v2:
        blk = blk[:old_v2.start()] + note + blk[old_v2.end():]
    elif old_v1:
        blk = blk[:old_v1.start()] + note + blk[old_v1.end():]
    else:
        anchor = f"\n      --kv-vram-cells {n}\n"
        blk = blk.replace(anchor, anchor + note, 1)
    for flag in ("\n      -np 3\n", "\n      --kv-unified\n", "\n      --spec-draft-window 16384\n"):
        if flag not in blk:
            raise ValueError(f"bonsai: {flag.strip()} missing -- the v1 layout (bench/deploy_kv_rank.py) comes first")
    c = c[:i0] + blk + c[i1:]
    if f'\n  "{VISION_ID}":\n' not in c:
        if VISION_BANNER_OLD in c:
            c = c.replace(VISION_BANNER_OLD, VISION_ENTRY, 1)
        else:
            anchor = "  # =====================================================================\n  # IMAGE GENERATION"
            if c.count(anchor) != 1:
                raise ValueError("config.yaml: no place for the vision entry (the IMAGE GENERATION banner)")
            c = c.replace(anchor, VISION_ENTRY + anchor, 1)
    if '\n        "ondemand":\n' not in c:
        anchor = '        "decision":\n          swap: false\n          exclusive: false\n          members:\n' \
                 '            - "clm-encoder"\n\n'
        if c.count(anchor) != 1:
            raise ValueError("config.yaml: the `decision` group not found once")
        c = c.replace(anchor, anchor + ONDEMAND_GROUP, 1)
    return c.replace("\n", nl)


def envs_for(n: int, lane: int = LANE) -> dict:
    """The proxy's values: the main cap (the line less the lane) and the lane."""
    return {"YAMADORI_MAIN_CAP": str(n - lane), "YAMADORI_LANE_TOKENS": str(lane)}


def edit_bat(text: str, envs: dict) -> str:
    """start-stack.bat: the old kv-layout block (and an earlier v2 block) out, this one in, before the
    first Python service starts."""
    bat, nl = _lf(text)
    for mark in OLD_MARKS + (MARK,):
        bat = re.sub(rf"REM {re.escape(mark)}\n(set \"YAMADORI_[A-Z_]+=\d+\"\n)+", "", bat)
    anchor = 'start "" /B "%PY%" "%CD%\\mcp\\tools_api.py"'
    if bat.count(anchor) != 1:
        raise ValueError("start-stack.bat: the tools_api start line not found once")
    block = f"REM {MARK}\n" + "".join(f'set "{k}={v}"\n' for k, v in envs.items())
    return bat.replace(anchor, block + anchor, 1).replace("\n", nl)


def edit_watchdog(text: str, envs: dict) -> str:
    """watchdog.ps1: the same values for a watchdog restart of the proxy, the tools API and the worker, as
    the `Env2` table deploy_kv_rank added (its restart loop already applies Env2 after Env)."""
    ps, nl = _lf(text)
    for mark in OLD_MARKS + (MARK,):
        ps = re.sub(r"\n +# " + re.escape(mark) + r"\n +Env2 = @\{[^}]*\}", "", ps)
    pairs = "; ".join(f"{k} = '{v}'" for k, v in envs.items())
    for match in ("Match = 'mcp[\\\\/]server\\.py'", "Match = 'tools_api\\.py'",
                  "Match = 'mcp[\\\\/]worker\\.py'"):
        if ps.count(match) != 1:
            raise ValueError(f"watchdog.ps1: {match} not found once")
        ps = ps.replace(match, match + f"\n       # {MARK}\n       Env2 = @{{ {pairs} }}", 1)
    if "$svc.Env2" not in ps:
        raise ValueError("watchdog.ps1: no Env2 loop (bench/deploy_kv_rank.py adds it; the v1 layout comes first)")
    return ps.replace("\n", nl)


def intended_props(n: int, lane: int = LANE) -> dict:
    """What the served /props should say after the restart."""
    return {"n_ctx": min(2 * n + lane, TRAINED_WINDOW), "total_slots": 3, "kv_vram_cells": n,
            "modalities": {"vision": False}}


def edit_fixture(text: str, served: dict) -> str:
    fx = json.loads(text)
    fx.update(served, recorded=time.strftime("%Y-%m-%d") + " (bench/deploy_layout_v2.py)")
    fx["_what"] = re.sub(r"modalities\.vision: true because config\.yaml launches `bonsai` with --mmproj\.",
                         "modalities.vision: false since layout v2 (bench/deploy_layout_v2.py): `bonsai` runs "
                         "without --mmproj and vision is `bonsai-vision` on the A4000.", fx["_what"])
    fx["_what"] = re.sub(r"PLANNED LAYOUT, NOT YET SERVED[^.]*\.",
                         "RECORDED from the served /props by bench/deploy_layout_v2.py.", fx["_what"])
    return json.dumps(fx, indent=2) + "\n"


def planned(n: int) -> dict:
    """{file: new text} for every file the deploy edits before the restart."""
    old = {f: read(f) for f in FILES}
    envs = envs_for(n)
    return {"config.yaml": edit_config(old["config.yaml"], n),
            "scripts/start-stack.bat": edit_bat(old["scripts/start-stack.bat"], envs),
            "scripts/watchdog.ps1": edit_watchdog(old["scripts/watchdog.ps1"], envs)}, old


def check_argv(argv_before: list[str], argv_after: list[str], env_before: dict, env_after: dict,
               n: int) -> None:
    """`bonsai`'s argv after the edit: -c, --kv-vram-cells, no --mmproj, the rest unchanged."""
    if env_after != env_before:
        raise ValueError("bonsai env changed")
    if argv_after[0] != argv_before[0]:
        raise ValueError(f"bonsai's binary changed: {argv_after[0]}")

    def val(av, flag):
        return av[av.index(flag) + 1] if flag in av else None
    want = {"-c": str(2 * n + LANE), "--kv-vram-cells": str(n), "--mmproj": None, "-np": "3",
            "--spec-draft-window": "16384"}
    got = {k: val(argv_after, k) for k in want}
    if got != want:
        raise ValueError(f"argv: {got} != {want}")
    if "--kv-unified" not in argv_after:
        raise ValueError("argv: --kv-unified missing")

    def rest(av):
        out, skip = [], False
        for x in av[1:]:
            if skip:
                skip = False
                continue
            if x in ("-c", "--kv-vram-cells", "--mmproj"):
                skip = True
                continue
            out.append(x)
        return out
    if rest(argv_after) != rest(argv_before):
        raise ValueError("bonsai argv changed beyond -c, --kv-vram-cells and --mmproj")


def check_vision_entry(config_text: str) -> dict:
    """The restored entry: on the A4000 by UUID, with the projector, in `ondemand`, sized in gpu_room."""
    import yaml
    import gpu_room
    cfg = yaml.safe_load(config_text)
    m = (cfg.get("models") or {}).get(VISION_ID) or {}
    cmd = str(m.get("cmd") or "")
    groups = (((cfg.get("routing") or {}).get("router") or {}).get("settings") or {}).get("groups") or {}
    rec = {"entry": bool(m), "a4000": any(A4000_UUID in str(e) for e in m.get("env") or []),
           "mmproj": f"--mmproj ${{models}}/{MMPROJ}" in cmd, "trunk": f"${{models}}/{VISION_TRUNK}" in cmd,
           "ondemand": VISION_ID in ((groups.get("ondemand") or {}).get("members") or []),
           "gpu_room_size": VISION_ID in gpu_room.SIZES,
           # the argv, not the comments (llama-swap drops comment lines too)
           "main_has_mmproj": any(ln.strip().startswith("--mmproj") for ln in str(
               ((cfg.get("models") or {}).get("bonsai") or {}).get("cmd")).splitlines())}
    bad = [k for k, v in rec.items() if (not v if k != "main_has_mmproj" else v)]
    if bad:
        raise ValueError(f"vision entry: {bad} ({rec})")
    return rec


# ------------------------------------------------------------------- main --
PS_RESTART = r"""
# every start-stack.bat cmd.exe first: a running batch file re-reads itself by byte offset, so a launcher alive from
# before this deploy edited start-stack.bat re-ran its tail and started a proxy WITHOUT the new environment lines
# (2026-09-28 14:14, the flash-next deploy)
Get-CimInstance Win32_Process -Filter "Name='cmd.exe'" | Where-Object { $_.CommandLine -like '*start-stack.bat*' } |
  ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
Get-Process llama-swap, llama-server -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
  Where-Object { $_.CommandLine -match 'mcp[\\/]server\.py|tools_api\.py|mcp[\\/]worker\.py|searx\.webapp' } |
  ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
Get-CimInstance Win32_Process -Filter "Name='wscript.exe'" -ErrorAction SilentlyContinue |
  Where-Object { $_.CommandLine -like '*start-stack-hidden*' } |
  ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
try { Stop-ScheduledTask -TaskName '__STACK__' -ErrorAction SilentlyContinue } catch {}
Start-Sleep -Seconds 6
Start-ScheduledTask -TaskName '__STACK__'
""".replace("__STACK__", STACK_TASK)

# The watchdog runs as its own Scheduled Task (scripts/install-autostart.ps1, every 5 minutes, MultipleInstances
# IgnoreNew): an instance alive from before the deploy keeps the watchdog.ps1 it started with (and its Env2
# values). Stopped BEFORE the restart (it must not restart a stack that is loading) and started after health.
PS_WATCHDOG_STOP = f"""
try {{ Stop-ScheduledTask -TaskName '{WATCHDOG_TASK}' -ErrorAction SilentlyContinue }} catch {{}}
Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" -ErrorAction SilentlyContinue |
  Where-Object {{ $_.CommandLine -like '*watchdog.ps1*' }} |
  ForEach-Object {{ Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }}
"""
PS_WATCHDOG_START = f"""
Start-ScheduledTask -TaskName '{WATCHDOG_TASK}'
(Get-ScheduledTask -TaskName '{WATCHDOG_TASK}').State
"""


def _ps(script: str, timeout: int = 180) -> str:
    r = subprocess.run(["powershell", "-NoProfile", "-Command", script], capture_output=True, text=True,
                       timeout=timeout, check=False)
    return (r.stdout or "").strip()[-400:]


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--line", type=int, help="N, the VRAM line bench/kv_rank.py (--layout v2) measured")
    ap.add_argument("--dry", action="store_true", help="make and check the edits, then put everything back")
    ap.add_argument("--preview", action="store_true", help="print every edit as a full diff; write nothing")
    a = ap.parse_args(argv)
    if not a.line:
        print("refusing: --line N is required (the VRAM line `python bench/kv_rank.py --window --layout v2` "
              "measured)")
        return 2
    n = a.line
    if n % 256 or n <= LANE:
        print(f"refusing: --line {n} is not a positive multiple of 256 above the lane ({LANE})")
        return 2
    new, old = planned(n)
    served_plan = intended_props(n)
    summary = {"line": n, "lane": LANE, "main_cap": n - LANE, "-c": 2 * n + LANE,
               "env": envs_for(n), "intended_props": served_plan}
    if a.preview:
        for f, t in new.items():
            sys.stdout.writelines(difflib.unified_diff(_lf(old[f])[0].splitlines(True),
                                                       _lf(t)[0].splitlines(True), f"a/{f}", f"b/{f}", n=3))
        fx = edit_fixture(old["mcp/fixtures/bonsai_props.json"], served_plan)
        print("\n# mcp/fixtures/bonsai_props.json: re-recorded from the SERVED /props after the restart; if it "
              "serves what is intended, the diff is:")
        sys.stdout.writelines(difflib.unified_diff(old["mcp/fixtures/bonsai_props.json"].splitlines(True),
                                                   fx.splitlines(True), "a/mcp/fixtures/bonsai_props.json",
                                                   "b/mcp/fixtures/bonsai_props.json", n=3))
        print("\n# the restart: kill the start-stack.bat launchers, stop the watchdog task, stop every service, "
              f"Start-ScheduledTask {STACK_TASK}, wait for health, Start-ScheduledTask {WATCHDOG_TASK}")
        print(json.dumps(summary, indent=1))
        return 0

    os.chdir(ROOT)
    import build_engine as be
    import engine_corruption as ec
    import verify_artifacts as va
    stamp = time.strftime("%Y%m%d-%H%M%S")
    backups = {}
    for f in FILES:
        backups[f] = f"{f}.bak-{stamp}"
        shutil.copyfile(f, backups[f])

    def rollback(why):
        for f, b in backups.items():
            shutil.copyfile(b, f)
        print(json.dumps({"verdict": "ROLLED BACK before restart", "why": why}, indent=1))
        sys.exit(1)

    before = {str(x) for x in va.verify("config.yaml") if x.severity == "error"}
    argv_before, env_before = ec.production_command()
    try:
        for f, t in new.items():
            write(f, t)
        argv_after, env_after = ec.production_command()
        check_argv(argv_before, argv_after, env_before, env_after, n)
        vision = check_vision_entry(read("config.yaml"))
        if edit_config(read("config.yaml"), n) != read("config.yaml"):
            raise ValueError("edit_config is not idempotent")
    except Exception as e:                                           # noqa: BLE001
        rollback(f"{type(e).__name__}: {e}")
    problems, ok = be.verify_deploy()
    errs = [str(x) for x in va.verify("config.yaml") if x.severity == "error" and str(x) not in before]
    print(json.dumps({"engines": problems, "engines_ok": ok, "models_new_errors": errs, "backups": backups,
                      "vision": vision, **summary}, indent=1))
    if problems or errs:
        rollback("checks failed")
    if a.dry:
        rollback("dry run: checks passed, nothing restarted")

    rec: dict = {"watchdog_stop": _ps(PS_WATCHDOG_STOP, 60)}
    _ps(PS_RESTART)
    want_up = {"llama-swap": "http://127.0.0.1:11434/health", "proxy": "http://127.0.0.1:1234/health",
               "tools-api": "http://127.0.0.1:1235/health", "searxng": "http://127.0.0.1:8888/healthz",
               "bonsai": "http://127.0.0.1:11434/upstream/bonsai/health"}
    t0, up = time.time(), {}
    while time.time() - t0 < 900:
        for k, u in want_up.items():
            if not up.get(k):
                try:
                    with urllib.request.urlopen(u, timeout=30) as r:
                        up[k] = r.status == 200
                except Exception:                                    # noqa: BLE001
                    up[k] = False
        if all(up.values()):
            break
        time.sleep(5)
    rec.update(health=up, seconds=round(time.time() - t0))
    rec["watchdog_start"] = _ps(PS_WATCHDOG_START, 60)
    try:
        with urllib.request.urlopen("http://127.0.0.1:11434/upstream/bonsai/props", timeout=60) as r:
            live = json.load(r)
        served = {"n_ctx": (live.get("default_generation_settings") or {}).get("n_ctx"),
                  "total_slots": live.get("total_slots"), "kv_vram_cells": live.get("kv_vram_cells"),
                  "modalities": {"vision": bool((live.get("modalities") or {}).get("vision"))}}
        rec.update(served=served, intended=served_plan, as_intended=served == served_plan)
        write("mcp/fixtures/bonsai_props.json", edit_fixture(read("mcp/fixtures/bonsai_props.json"), served))
    except Exception as e:                                           # noqa: BLE001
        rec["fixture"] = f"not re-recorded: {type(e).__name__}: {e}"
    rec["next"] = ("python scripts/deploy_check.py --key-file PATH (a deploy is not good until it exits 0); "
                   "vision loads on the A4000 on the first image (mcp/test_live_stack.py --only images)")
    print(json.dumps(rec, indent=1))
    return 0 if all(up.values()) and rec.get("as_intended") else 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
