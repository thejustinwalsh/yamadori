"""Deploy LAYOUT V3 for `bonsai`: ONE conversation (operator, 2026-09-29, verbatim: "we should not have a second
conversation at all, it is too slow, we have a second gpu if we want a second conversation, that is how it has to
play out, the jjava engine should be the only other thing we need ready to go"), with the line
`python bench/kv_rank.py --window --layout v3` measured.

-np 1 IS THE DEFAULT (operator, 2026-09-30, "2. Yes": Bonsai's jjava AND side calls move to bonsai-a4000, like
Flash-Next; the main card -np 1, no lane). Evidence: bench/kv_rank.py --layout v3 lane3 (n=3 per arm), a jjava
burst on the lane cut main's decode ~30% (8K 73.8 -> 51.3 tok/s, 32K 72.7 -> 52.4, 64K 73.0 -> 51.0). So -c N,
--kv-vram-cells N, the main cap N (no lane), YAMADORI_LANE_TOKENS 0. THE LINE AT -np 1: N = 209,920 was measured
at -np 2 (bench/results/kv_rank/20260930-v3, one fit); one slot fewer frees only that slot's per-slot buffers, so
the -np 2 line is SAFE at -np 1 (derived: the unified KV pool is -c cells either way) and somewhat below the -np 1
maximum -- a short -np 1 fit may raise it later. The served /props is checked after the restart (`as_intended`).
PRECONDITION: bench/deploy_tier_models.py ran first (bonsai-a4000 in config.yaml and YAMADORI_TIER_MODELS set):
with no lane, jjava and side calls have nowhere else to go. `--np 2` keeps the 2026-09-29 form (the conversation +
the lane) for a comparison. In the style of bench/deploy_layout_v2.py (whose restart,
watchdog handling and fixture re-record it reuses): every file is backed up and put back if a check fails before
the restart; the start-stack.bat launchers are killed first; the watchdog task is stopped and started again; the
served /props is re-recorded into the offline fixture. A deploy is not good until scripts/deploy_check.py exits 0.

WHAT CHANGES (only in `bonsai`'s argv and the proxy's env; layout v2's other edits -- no --mmproj, bonsai-vision
and its `ondemand` group -- are kept and checked):

  config.yaml, `bonsai`   -np 3 -> -np 2 (slot 0 the conversation, slot 1 the lane); -c 2N + lane -> -c N (every
                          cell in VRAM: the cells above N were host RAM for the second conversation, ~6 GB pinned);
                          --kv-vram-cells N (= -c: nothing tiers). Nothing else in its argv or env changes (checked).
  scripts/start-stack.bat, scripts/watchdog.ps1
                          the v2 block is REPLACED: YAMADORI_MAIN_CAP = N - LANE (budget.main_cap: the line less
                          the lane) and YAMADORI_LANE_TOKENS = LANE, for the proxy, the tools API and the worker.
  mcp/fixtures/bonsai_props.json
                          re-recorded from the served /props: n_ctx = min(N, 262,144), total_slots 2, kv_vram_cells N.

The proxy side is already in the working tree and turns on by itself when /props says 2 slots: mcp/slots.py ONE
CONVERSATION (a second conversation is refused 503 conversation_at_capacity until bonsai-a4000 serves it) and the
lane CLEARED after each burst (slots LANE_KEEP False).

    python bench/deploy_layout_v3.py --line N --preview
    python bench/deploy_layout_v3.py --line N --dry
    python bench/deploy_layout_v3.py --line N
    python bench/deploy_layout_v3.py --selftest
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import shutil
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "mcp"))
import budget                                                        # noqa: E402
import deploy_layout_v2 as dl                                        # noqa: E402

FILES = ["config.yaml", "scripts/start-stack.bat", "scripts/watchdog.ps1", "mcp/fixtures/bonsai_props.json"]
MARK = "layout v3 (bench/deploy_layout_v3.py)"
OLD_MARKS = (dl.MARK,) + dl.OLD_MARKS
LANE = budget.LANE_TOKENS
NP = 1                    # --np (set by main()); 1 = the operator's 2026-09-30 decision, 2 = the lane form
TRAINED_WINDOW = dl.TRAINED_WINDOW
read, write, _lf = dl.read, dl.write, dl._lf


def edit_config(text: str, n: int) -> str:
    """config.yaml's `bonsai` for layout v3. Idempotent."""
    c, nl = _lf(text)
    i0, i1 = dl._bonsai_block(c)
    blk = c[i0:i1]
    for pat, new, what in ((r"\n      -c \d+\n", f"\n      -c {n}\n", "-c"),
                           (r"\n      --kv-vram-cells \d+\n", f"\n      --kv-vram-cells {n}\n", "--kv-vram-cells"),
                           (r"\n      -np [123]\n", f"\n      -np {NP}\n", "-np")):
        if len(re.findall(pat, blk)) != 1:
            raise ValueError(f"bonsai: {what} not found once")
        blk = re.sub(pat, new, blk)
    if re.search(r"\n      --mmproj ", blk):
        raise ValueError("bonsai still has --mmproj: layout v2 (bench/deploy_layout_v2.py) comes first")
    if NP == 1:
        note = (f"      # {MARK}, operator 2026-09-30 (\"2. Yes\": jjava and side calls on bonsai-a4000): -np 1 =\n"
                "      # ONE conversation, no lane (a jjava burst on the lane cut decode ~30%, bench/kv_rank.py\n"
                f"      # --layout v3 lane3, n=3); -c = N = {n}, every cell in VRAM (bench/kv_rank.py --layout v3,\n"
                "      # measured at -np 2: safe at -np 1); the main cap N (budget.main_cap, YAMADORI_LANE_TOKENS 0)\n")
    else:
        note = (f"      # {MARK}, operator 2026-09-29 (\"we should not have a second conversation at all, it is too\n"
                "      # slow, we have a second gpu if we want a second conversation\"): 2 slots = ONE conversation +\n"
                f"      # the jjava lane (budget.LANE_TOKENS = {LANE}, cleared after each burst); -c = N = {n}, every cell\n"
                f"      # in VRAM (bench/kv_rank.py --layout v3); the main cap N - the lane (budget.main_cap)\n")
    old = re.search(r"      # " + re.escape(MARK) + r", operator 2026-09-(?:29|30)[^\n]*\n(?:      #[^\n]*\n){3}", blk)
    older = re.search(r"      # " + re.escape(dl.MARK) + r", operator 2026-09-29 \(\"the point[^\n]*\n"
                      r"(?:      #[^\n]*\n){5}", blk)
    if old:
        blk = blk[:old.start()] + note + blk[old.end():]
    elif older:
        blk = blk[:older.start()] + note + blk[older.end():]
    else:
        anchor = f"\n      --kv-vram-cells {n}\n"
        blk = blk.replace(anchor, anchor + note, 1)
    return (c[:i0] + blk + c[i1:]).replace("\n", nl)


def lane_of() -> int:
    """The lane's cells: none at -np 1 (jjava and side calls on bonsai-a4000)."""
    return 0 if NP == 1 else LANE


def envs_for(n: int) -> dict:
    # YAMADORI_SLOTS (2026-10-02): the served slot count, DECLARED -- a live /props read loses the race when a tier
    # swap takes the main model off the card seconds after a restart (mcp/slots.py reads it; mcp/budget.py takes
    # an unread pool from YAMADORI_MAIN_CAP + the lane)
    return {"YAMADORI_MAIN_CAP": str(n - lane_of()), "YAMADORI_LANE_TOKENS": str(lane_of()),
            "YAMADORI_SLOTS": str(NP)}


def tier_deployed(cfg: str, bat: str) -> list[str]:
    """-np 1's precondition: bench/deploy_tier_models.py ran (bonsai-a4000 served, the table on). [] when met."""
    c = cfg.replace("\r\n", "\n")
    why = []
    if '\n  "bonsai-a4000":\n' not in c and "\n  bonsai-a4000:\n" not in c:
        why.append("no bonsai-a4000 in config.yaml: bench/deploy_tier_models.py comes first (with no lane, jjava and "
                   "side calls run there)")
    if "YAMADORI_TIER_MODELS=" not in bat:
        why.append("YAMADORI_TIER_MODELS not set in start-stack.bat: bench/deploy_tier_models.py comes first (the "
                   "table's bonsai row routes jjava and side calls to bonsai-a4000)")
    return why


def edit_bat(text: str, envs: dict) -> str:
    bat, nl = _lf(text)
    for mark in OLD_MARKS + (MARK,):
        bat = re.sub(rf"REM {re.escape(mark)}\n(set \"YAMADORI_[A-Z_]+=\d+\"\n)+", "", bat)
    anchor = 'start "" /B "%PY%" "%CD%\\mcp\\tools_api.py"'
    if bat.count(anchor) != 1:
        raise ValueError("start-stack.bat: the tools_api start line not found once")
    block = f"REM {MARK}\n" + "".join(f'set "{k}={v}"\n' for k, v in envs.items())
    return bat.replace(anchor, block + anchor, 1).replace("\n", nl)


def edit_watchdog(text: str, envs: dict) -> str:
    ps, nl = _lf(text)
    for mark in OLD_MARKS + (MARK,):
        ps = re.sub(r"\n +# " + re.escape(mark) + r"\n +Env2 = @\{[^}]*\}", "", ps)
    pairs = "; ".join(f"{k} = '{v}'" for k, v in envs.items())
    for match in ("Match = 'mcp[\\\\/]server\\.py'", "Match = 'tools_api\\.py'", "Match = 'mcp[\\\\/]worker\\.py'"):
        if ps.count(match) != 1:
            raise ValueError(f"watchdog.ps1: {match} not found once")
        ps = ps.replace(match, match + f"\n       # {MARK}\n       Env2 = @{{ {pairs} }}", 1)
    if "$svc.Env2" not in ps:
        raise ValueError("watchdog.ps1: no Env2 loop")
    return ps.replace("\n", nl)


def intended_props(n: int) -> dict:
    return {"n_ctx": min(n, TRAINED_WINDOW), "total_slots": NP, "kv_vram_cells": n, "modalities": {"vision": False}}


def check_argv(before: list[str], after: list[str], env_b: dict, env_a: dict, n: int) -> None:
    if env_a != env_b:
        raise ValueError("bonsai env changed")

    def val(av, flag):
        return av[av.index(flag) + 1] if flag in av else None
    want = {"-c": str(n), "--kv-vram-cells": str(n), "-np": str(NP)}
    got = {k: val(after, k) for k in want}
    if got != want or "--kv-unified" not in after or "--mmproj" in after:
        raise ValueError(f"argv: {got} != {want} (or --kv-unified missing / --mmproj present)")

    def rest(av):
        out, skip = [], False
        for x in av[1:]:
            if skip:
                skip = False
                continue
            if x in ("-c", "--kv-vram-cells", "-np"):
                skip = True
                continue
            out.append(x)
        return out
    if rest(after) != rest(before):
        raise ValueError("bonsai argv changed beyond -c, --kv-vram-cells and -np")


def planned(n: int) -> tuple[dict, dict]:
    old = {f: read(f) for f in FILES}
    envs = envs_for(n)
    return {"config.yaml": edit_config(old["config.yaml"], n),
            "scripts/start-stack.bat": edit_bat(old["scripts/start-stack.bat"], envs),
            "scripts/watchdog.ps1": edit_watchdog(old["scripts/watchdog.ps1"], envs)}, old


def selftest() -> int:
    global NP
    bad = 0

    def check(ok, what):
        nonlocal bad
        print(("ok    " if ok else "FAIL  ") + what)
        bad += 0 if ok else 1
    import yaml
    cfg = read("config.yaml")
    has_v2 = not re.search(r"\n      --mmproj ", dl._lf(cfg)[0][slice(*dl._bonsai_block(dl._lf(cfg)[0]))])
    if not has_v2:
        print("note: the live config.yaml predates layout v2; the edit is checked on a v2-shaped copy")
        cfg = dl.edit_config(cfg, 170496)
    saved = NP
    try:
        for np_ in (1, 2):
            NP = np_
            lane = lane_of()
            new = edit_config(cfg, 174592)
            y = yaml.safe_load(new)
            cmd = str(y["models"]["bonsai"]["cmd"])
            check(f"\n-np {np_}\n" in "\n" + "\n".join(x.strip() for x in cmd.splitlines()) + "\n"
                  and "-c 174592" in cmd and "--kv-vram-cells 174592" in cmd,
                  f"-np {np_}: bonsai -np {np_}, -c N, --kv-vram-cells N")
            check(edit_config(new, 174592) == new, f"-np {np_}: idempotent")
            yb = yaml.safe_load(cfg)
            for k in ("bonsai-vision", "flash-next"):
                check((yb["models"].get(k) == y["models"].get(k)), f"-np {np_}: {k} untouched")
            check(yb["routing"] == y["routing"], f"-np {np_}: the groups untouched")
            cap = 174592 - lane
            bat = edit_bat(read("scripts/start-stack.bat"), envs_for(174592))
            check(bat.count(f'set "YAMADORI_MAIN_CAP={cap}"') == 1
                  and bat.count(f'set "YAMADORI_LANE_TOKENS={lane}"') == 1
                  and "REM " + dl.MARK not in bat and edit_bat(bat, envs_for(174592)) == bat,
                  f"-np {np_}: start-stack.bat: the v2 block replaced, main cap = N - {lane}, idempotent")
            ps = edit_watchdog(read("scripts/watchdog.ps1"), envs_for(174592))
            check(ps.count(f"YAMADORI_MAIN_CAP = '{cap}'") == 3 and edit_watchdog(ps, envs_for(174592)) == ps,
                  f"-np {np_}: watchdog.ps1: three services, idempotent")
            check(intended_props(174592) == {"n_ctx": 174592, "total_slots": np_, "kv_vram_cells": 174592,
                                             "modalities": {"vision": False}}, f"-np {np_}: the intended /props")
        NP = 1
        one = edit_config(cfg, 174592)
        NP = 2
        two = edit_config(one, 174592)
        blk = two[slice(*dl._bonsai_block(dl._lf(two)[0]))] if "\r\n" not in two else two
        check(blk.count("# " + MARK) == 1 and "operator 2026-09-29" in blk and "operator 2026-09-30" not in blk,
              "the note is replaced when the form changes (-np 1 -> -np 2)")
        check(bool(tier_deployed("models:\n  bonsai:\n", "")) and not tier_deployed(
            'models:\n  "bonsai-a4000":\n', 'set "YAMADORI_TIER_MODELS=mcp\\tier_models.yaml"\n'),
            "-np 1 refuses until bench/deploy_tier_models.py ran")
    finally:
        NP = saved
    print(f"\n{'all passed' if not bad else f'{bad} FAILED'}")
    return 1 if bad else 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--line", type=int)
    ap.add_argument("--np", type=int, choices=(1, 2), default=1,
                    help="1 (default; operator 2026-09-30): no lane, jjava and side calls on bonsai-a4000; 2: the lane")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    os.chdir(ROOT)
    if a.selftest:
        return selftest()
    global NP
    NP = a.np
    if not a.line or a.line % 256 or a.line <= lane_of():
        print("refusing: --line N (bench/kv_rank.py --window --layout v3), a multiple of 256 above the lane")
        return 2
    if NP == 1:
        why = tier_deployed(read("config.yaml"), read("scripts/start-stack.bat"))
        if why:
            print(json.dumps({"verdict": "REFUSED", "why": why}, indent=1))
            return 2
    n = a.line
    new, old = planned(n)
    if a.preview:
        for f, t in new.items():
            sys.stdout.writelines(difflib.unified_diff(_lf(old[f])[0].splitlines(True), _lf(t)[0].splitlines(True),
                                                       f"a/{f}", f"b/{f}", n=3))
        print(json.dumps({"line": n, "np": NP, "main_cap": n - lane_of(), "-c": n, "env": envs_for(n),
                          "intended_props": intended_props(n)}, indent=1))
        return 0
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
    argv_b, env_b = ec.production_command()
    try:
        for f, t in new.items():
            write(f, t)
        argv_a, env_a = ec.production_command()
        check_argv(argv_b, argv_a, env_b, env_a, n)
        dl.check_vision_entry(read("config.yaml"))
    except Exception as e:                                           # noqa: BLE001
        rollback(f"{type(e).__name__}: {e}")
    problems, ok = be.verify_deploy()
    errs = [str(x) for x in va.verify("config.yaml") if x.severity == "error" and str(x) not in before]
    if problems or errs:
        rollback(f"checks failed: {problems or errs}")
    if a.dry:
        rollback("dry run: checks passed, nothing restarted")
    rec: dict = {"watchdog_stop": dl._ps(dl.PS_WATCHDOG_STOP, 60)}
    dl._ps(dl.PS_RESTART)
    want_up = {"llama-swap": "http://127.0.0.1:11434/health", "proxy": "http://127.0.0.1:1234/health",
               "tools-api": "http://127.0.0.1:1235/health", "bonsai": "http://127.0.0.1:11434/upstream/bonsai/health"}
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
    rec["watchdog_start"] = dl._ps(dl.PS_WATCHDOG_START, 60)
    try:
        with urllib.request.urlopen("http://127.0.0.1:11434/upstream/bonsai/props", timeout=60) as r:
            live = json.load(r)
        served = {"n_ctx": (live.get("default_generation_settings") or {}).get("n_ctx"),
                  "total_slots": live.get("total_slots"), "kv_vram_cells": live.get("kv_vram_cells"),
                  "modalities": {"vision": bool((live.get("modalities") or {}).get("vision"))}}
        rec.update(served=served, intended=intended_props(n), as_intended=served == intended_props(n))
        write("mcp/fixtures/bonsai_props.json", dl.edit_fixture(read("mcp/fixtures/bonsai_props.json"), served))
    except Exception as e:                                           # noqa: BLE001
        rec["fixture"] = f"not re-recorded: {type(e).__name__}: {e}"
    rec["next"] = "python scripts/deploy_check.py --key-file PATH"
    print(json.dumps(rec, indent=1))
    return 0 if all(up.values()) and rec.get("as_intended") else 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
