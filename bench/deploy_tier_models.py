"""Deploy ONE MODEL PER EFFORT TIER (operator, 2026-09-29: "mirai xhigh flash-next max, and bonsai everything else"),
AFTER Mirai S's GPU gate (bench/mirai_s_gate.py) passed. In the style of bench/deploy_flash_next.py and
bench/deploy_layout_v2.py: every file it touches is backed up and put back if a check fails before the restart; the
old start-stack.bat launchers are killed (a running batch file re-reads itself by byte offset: 2026-09-28 14:14), the
watchdog task is stopped, ALL services restart through the Scheduled Task, this waits for health, and the watchdog
task is started again. The deploy is not good until `python scripts/deploy_check.py --key-file PATH` exits 0.

WHAT CHANGES (and what it keeps)

  config.yaml            the macro `server_mirai_s` (engines/manifest.yaml llama-mirai-s, its shipped binary); the
                         `mirai-s` entry from config.mirai-s.fragment.yaml with the gate's window ({{CTX}}), MTP when
                         the gate's `mtp` arm passed its corruption checks, and --mmproj only with --mirai-mmproj (by
                         default vision stays off the main card, layout v2's rule, and images go to `bonsai-vision`
                         on the A4000, which must then exist); the `primary` group gains `mirai-s` (swap: true, as
                         the flash-next deploy left it). NOTHING ELSE: `bonsai`, `bonsai-vision`, the `ondemand`
                         group and `flash-next` are checked byte-identical before and after (the layout-v2 deploy's
                         edits and the flash-next agent's engine stay as they are).
  mcp/tier_models.yaml   mirai-s's `window` {ctx, slots 3, main_cap = ctx - budget.LANE_TOKENS} (layout v2's rule,
                         the line = the whole pool on an untiered engine) and `vision` (whether its entry has
                         --mmproj). The table's profiles are not touched.
  scripts/start-stack.bat, scripts/watchdog.ps1
                         YAMADORI_TIER_MODELS=mcp\\tier_models.yaml for the proxy, the tools API and the worker
                         (`EnvTier` in the watchdog, applied after its other tables). YAMADORI_MAX_MODEL stays: the
                         table wins over it (mcp/tier_models.py), and a rollback of this deploy leaves max mode as
                         it was.
  engines/manifest.yaml  llama-mirai-s: config_refs [mirai-s].
  models/manifest.yaml   mirai-s-qwen3.8-27b-gguf (and mirai-s-mmproj with --mirai-mmproj): in_service,
                         config_entries [mirai-s].

    python bench/deploy_tier_models.py --gate DIR --preview   # every edit as a diff; nothing is written
    python bench/deploy_tier_models.py --gate DIR --dry       # write, check, put everything back
    python bench/deploy_tier_models.py --gate DIR             # deploy (restarts the stack)
    python bench/deploy_tier_models.py --selftest             # the pure edits, offline

The edits are pure functions of the files' text (edit_config, edit_table, edit_bat, edit_watchdog, edit_engines,
edit_models).
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

ENGINE = "llama-mirai-s"
MODEL_ID = "mirai-s"
FRAGMENT = "config.mirai-s.fragment.yaml"
TABLE = "mcp/tier_models.yaml"
FILES = ["config.yaml", TABLE, "scripts/start-stack.bat", "scripts/watchdog.ps1", "engines/manifest.yaml",
         "models/manifest.yaml"]
MARK = "tier models (bench/deploy_tier_models.py)"
ENVS = {"YAMADORI_TIER_MODELS": "mcp\\tier_models.yaml"}
MODEL_FILES = ["mirai-s-qwen3.8-27b-gguf"]
MMPROJ_FILE = "mirai-s-mmproj"
SLOTS = 3
# the entries this deploy must leave byte-identical (other deploys own them)
KEEP = ("bonsai", "bonsai-vision", "flash-next")


def read(f: str) -> str:
    return open(os.path.join(ROOT, f), encoding="utf-8", newline="").read()


def write(f: str, s: str) -> None:
    open(os.path.join(ROOT, f), "w", encoding="utf-8", newline="").write(s)


def _lf(s: str) -> tuple[str, str]:
    return s.replace("\r\n", "\n"), ("\r\n" if "\r\n" in s else "\n")


def lane() -> int:
    import budget
    return int(budget.LANE_TOKENS)


# ------------------------------------------------------------------ values --
def values_from_gate(gate: dict, over: dict) -> tuple[dict, list[str]]:
    """{ctx, mtp, mmproj} and the reasons the gate does NOT allow a deploy (empty = allowed)."""
    why: list[str] = []
    if not (gate.get("kernels") or {}).get("PASS"):
        why.append("kernels: test-backend-ops did not pass (or did not run)")
    if not (gate.get("api") or {}).get("PASS"):
        why.append("api: the llama-server contract the proxy and the decider need did not pass")
    arms = gate.get("arms") or {}
    base = arms.get("base") or {}
    corr = base.get("corrupt") or {}
    if not corr or corr.get("error") or corr.get("guard"):
        why.append("base: the corruption checks did not run clean")
    if not (base.get("needles") or {}).get("PASS"):
        why.append("base: needles did not pass 5/5 at every depth")
    mmproj = bool(over.get("mmproj"))
    if gate.get("mmproj", True) != mmproj:
        why.append(f"the gate ran with mmproj={gate.get('mmproj', True)} but this deploy asks mmproj={mmproj}: the "
                   f"fit differs (re-run the gate {'without' if not mmproj else 'with'} --no-mmproj)")
    ctx = over.get("ctx") or gate.get("ctx_used")
    if not ctx:
        why.append("no window: the gate's fit did not run and --ctx was not given")
    mc = (arms.get("mtp") or {}).get("corrupt") or {}
    mtp = bool(mc) and not mc.get("error") and not mc.get("guard") and over.get("mtp", True)
    return {"ctx": int(ctx or 0), "mtp": mtp, "mmproj": mmproj}, why


# ------------------------------------------------------------------ edits --
def fragment(text: str, values: dict, engine_bin: str) -> tuple[str, str]:
    """(the macro line, the model block) from config.mirai-s.fragment.yaml."""
    f, _ = _lf(text)
    f = f.replace("{{CTX}}", str(values["ctx"]))
    left = re.findall(r"\{\{([A-Z_]+)\}\}", f)
    if left:
        raise ValueError(f"placeholders not filled: {left}")
    macro = f'  server_mirai_s: "{engine_bin}"'
    block = f[f.index('\n  "mirai-s":\n') + 1:]
    block = block[:block.index("\n# groups:")] if "\n# groups:" in block else block
    block = re.sub(r" +$", "", block, flags=re.M).rstrip("\n") + "\n"
    if values.get("mtp"):
        block = block.replace("      # --spec-type draft-mtp\n", "      --spec-type draft-mtp\n")
        block = block.replace("      # --spec-draft-n-max 3\n", "      --spec-draft-n-max 3\n")
    if not values.get("mmproj"):
        block = re.sub(r"\n      --mmproj [^\n]+\n",
                       f"\n      # {MARK}: no --mmproj -- vision off the main card (layout v2, operator 2026-09-29:\n"
                       "      # \"Vision can go to second card and swap in and out\"); images go to bonsai-vision.\n",
                       block, count=1)
    block = block.replace('name: "Mirai S Qwen3.8-27B 2.4-bit (5060 Ti) -- CANDIDATE"',
                          'name: "Mirai S Qwen3.8-27B 2.4-bit (5060 Ti) -- the xhigh tier"')
    return macro, block


def _entry(c: str, mid: str) -> str | None:
    i0 = c.find(f'\n  "{mid}":\n')
    if i0 < 0:
        return None
    m = re.search(r'\n  "[^"]+":\n|\n[a-z]', c[i0 + 5:])
    return c[i0: i0 + 5 + m.start()] if m else c[i0:]


def edit_config(text: str, macro: str, block: str) -> str:
    c, nl = _lf(text)
    if f'\n  "{MODEL_ID}":\n' in c:
        raise ValueError(f"config.yaml already has a {MODEL_ID} entry")
    m = re.search(r"^  server_nudge: .*$", c, re.M)
    if not m:
        raise ValueError("server_nudge macro not found")
    c = c[:m.end()] + "\n" + f"  # {MARK}\n" + macro + c[m.end():]
    i0 = c.index('\n  "bonsai":\n')
    i1 = re.search(r'\n  "[^"]+":\n', c[i0 + 5:]).start() + i0 + 5
    c = c[:i1] + "\n\n" + f"  # {MARK}\n" + block.rstrip("\n") + c[i1:]
    g0 = c.index('\n        "primary":\n')
    g1 = c.index("\n\n", g0)
    grp = c[g0:g1]
    if "          swap: true" not in grp or '            - "bonsai"' not in grp:
        raise ValueError("the primary group is not swap: true with bonsai (the flash-next deploy comes first)")
    grp = grp.replace('            - "bonsai"\n', '            - "bonsai"\n'
                      f'            # {MARK}\n            - "{MODEL_ID}"\n', 1)
    return (c[:g0] + grp + c[g1:]).replace("\n", nl)


def edit_table(text: str, ctx: int, vision: bool, lane_tokens: int) -> str:
    """mirai-s's window and vision in mcp/tier_models.yaml (the rest byte-identical)."""
    t, nl = _lf(text)
    i0 = t.index("\n  mirai-s:\n")
    i1 = t.index("\n  flash-next:\n", i0)
    blk = t[i0:i1]
    win = (f"\n    window:\n      ctx: {ctx}\n      slots: {SLOTS}\n      main_cap: {ctx - lane_tokens}\n"
           f"      source: >-\n        derived: config.yaml `mirai-s` -c {ctx} (bench/mirai_s_gate.py's fit) -np {SLOTS}"
           f" --kv-unified,\n        written by {MARK}; the main cap is layout v2's rule (the line less the decider"
           f" lane,\n        budget.LANE_TOKENS {lane_tokens}) with the line = the whole pool (no tiered KV cache on"
           f" this engine)\n")
    new, n = re.subn(r"\n    window: [^\n]*\n|\n    window:\n(?:      [^\n]*\n|        [^\n]*\n)+", win, blk, count=1)
    if n != 1:
        raise ValueError("mirai-s: no window line in the table")
    new, n = re.subn(r"\n    vision: [^\n]*\n", f"\n    vision: {'true' if vision else 'false'}\n", new, count=1)
    if n != 1:
        raise ValueError("mirai-s: no vision line in the table")
    return (t[:i0] + new + t[i1:]).replace("\n", nl)


def edit_bat(text: str) -> str:
    bat, nl = _lf(text)
    bat = re.sub(rf"REM {re.escape(MARK)}\n(set \"YAMADORI_TIER_MODELS=[^\"]*\"\n)", "", bat)
    anchor = 'start "" /B "%PY%" "%CD%\\mcp\\tools_api.py"'
    if bat.count(anchor) != 1:
        raise ValueError("start-stack.bat: the tools_api start line not found once")
    block = f"REM {MARK}\n" + "".join(f'set "{k}={v}"\n' for k, v in ENVS.items())
    return bat.replace(anchor, block + anchor, 1).replace("\n", nl)


def edit_watchdog(text: str) -> str:
    """The same variable for a watchdog restart of the proxy, the tools API and the worker: a hashtable `EnvTier`,
    applied after the service's other tables."""
    ps, nl = _lf(text)
    ps = re.sub(r"\n +# " + re.escape(MARK) + r"\n +EnvTier = @\{[^}]*\}", "", ps)
    pairs = "; ".join(f"{k} = '{v}'" for k, v in ENVS.items())
    for match in ("Match = 'mcp[\\\\/]server\\.py'", "Match = 'tools_api\\.py'", "Match = 'mcp[\\\\/]worker\\.py'"):
        if ps.count(match) != 1:
            raise ValueError(f"watchdog.ps1: {match} not found once")
        ps = ps.replace(match, match + f"\n       # {MARK}\n       EnvTier = @{{ {pairs} }}", 1)
    if "$svc.EnvTier" not in ps:
        old = "    if ($svc.Env) {\n"
        if ps.count(old) != 1:
            raise ValueError("watchdog.ps1: the Env loop not found once")
        ps = ps.replace(old, "    if ($svc.EnvTier) {\n"
                        "        foreach ($k in $svc.EnvTier.Keys) {\n"
                        "            [Environment]::SetEnvironmentVariable($k, $svc.EnvTier[$k], 'Process')\n"
                        "        }\n    }\n" + old, 1)
    return ps.replace("\n", nl)


def edit_engines(text: str) -> str:
    t, nl = _lf(text)
    i0 = t.index(f"\n  {ENGINE}:\n")
    nxt = re.search(r"\n  [A-Za-z0-9_.-]+:\n|\n  # ----", t[i0 + 1:])
    i1 = i0 + 1 + nxt.start() if nxt else len(t)
    blk = t[i0:i1]
    if "config_refs:" in blk:
        new = re.sub(r"\n    config_refs: \[[^\]]*\][^\n]*", f"\n    config_refs: [{MODEL_ID}]", blk, count=1)
    else:
        new = blk.replace("\n    kind: source\n", f"\n    kind: source\n    config_refs: [{MODEL_ID}]\n", 1)
    if new == blk and f"config_refs: [{MODEL_ID}]" not in blk:
        raise ValueError(f"{ENGINE}: no place for config_refs")
    return (t[:i0] + new + t[i1:]).replace("\n", nl)


def edit_models(text: str, mmproj: bool) -> str:
    t, nl = _lf(text)
    for aid in MODEL_FILES + ([MMPROJ_FILE] if mmproj else []):
        i0 = t.index(f"\n  - id: {aid}\n")
        m = re.search(r"\n  - id: |\n[a-z]", t[i0 + 5:])
        i1 = i0 + 5 + m.start() if m else len(t)
        blk = t[i0:i1]
        new = blk.replace("\n    status: candidate\n", "\n    status: in_service\n", 1)
        new = new.replace("\n    config_entries: []\n", f"\n    config_entries: [{MODEL_ID}]\n", 1)
        if new == blk:
            raise ValueError(f"{aid}: not a candidate with config_entries []")
        t = t[:i0] + new + t[i1:]
    return t.replace("\n", nl)


def untouched(before: str, after: str) -> list[str]:
    """The KEEP entries and every group but `primary` that changed -- must be none (compared as llama-swap reads
    them: parsed, comment lines dropped from each cmd)."""
    import yaml
    b, a = yaml.safe_load(before), yaml.safe_load(after)

    def norm(m):
        m = dict(m or {})
        if "cmd" in m:
            m["cmd"] = [ln.strip() for ln in str(m["cmd"]).splitlines()
                        if ln.strip() and not ln.strip().startswith("#")]
        return m
    bad = [k for k in KEEP if norm((b.get("models") or {}).get(k)) != norm((a.get("models") or {}).get(k))]
    gb = b["routing"]["router"]["settings"]["groups"]
    ga = a["routing"]["router"]["settings"]["groups"]
    bad += [f"group {g}" for g in set(gb) | set(ga) if g != "primary" and gb.get(g) != ga.get(g)]
    pb, pa = dict(gb["primary"]), dict(ga["primary"])
    if [m for m in pa.pop("members") if m != MODEL_ID] != pb.pop("members") or pa != pb:
        bad.append("group primary beyond the new member")
    return bad


def planned(values: dict, engine_bin: str) -> tuple[dict, dict]:
    old = {f: read(f) for f in FILES}
    macro, block = fragment(read(FRAGMENT), values, engine_bin)
    new = {"config.yaml": edit_config(old["config.yaml"], macro, block),
           TABLE: edit_table(old[TABLE], values["ctx"], values["mmproj"], lane()),
           "scripts/start-stack.bat": edit_bat(old["scripts/start-stack.bat"]),
           "scripts/watchdog.ps1": edit_watchdog(old["scripts/watchdog.ps1"]),
           "engines/manifest.yaml": edit_engines(old["engines/manifest.yaml"]),
           "models/manifest.yaml": edit_models(old["models/manifest.yaml"], values["mmproj"])}
    return new, old


# -------------------------------------------------------------- selftest --
def selftest() -> int:
    bad = 0

    def check(ok, what):
        nonlocal bad
        print(("ok    " if ok else "FAIL  ") + what)
        bad += 0 if ok else 1
    gate = {"kernels": {"PASS": True}, "api": {"PASS": True}, "ctx_used": 98304, "mmproj": False,
            "arms": {"base": {"corrupt": {"greedy": []}, "needles": {"PASS": True}},
                     "mtp": {"corrupt": {"greedy": []}}}}
    v, why = values_from_gate(gate, {})
    check(not why and v == {"ctx": 98304, "mtp": True, "mmproj": False}, "the gate's window, MTP, no mmproj")
    _, why2 = values_from_gate(gate, {"mmproj": True})
    check(any("mmproj" in w for w in why2), "a deploy that differs from the gate's mmproj arm is refused")
    _, why3 = values_from_gate({}, {})
    check(len(why3) >= 4, "no gate, no deploy")
    macro, block = fragment(read(FRAGMENT), v, "C:/x/llama-server.exe")
    check("-c 98304" in block and "\n      --spec-type draft-mtp\n" in block and "\n      --mmproj" not in block
          and "{{" not in block and block.startswith('  "mirai-s":') and "# groups:" not in block,
          "the fragment fills: the window, MTP on, no projector")
    cfg = read("config.yaml")
    new = edit_config(cfg, macro, block)
    import yaml
    y = yaml.safe_load(new)
    grp = y["routing"]["router"]["settings"]["groups"]["primary"]
    check(grp["swap"] is True and "mirai-s" in grp["members"] and grp["members"][0] == "bonsai",
          "primary gains mirai-s, swap kept", )
    check(not untouched(cfg, new), "bonsai, bonsai-vision, flash-next and the ondemand group are byte-identical")
    mm = y["models"]["mirai-s"]
    check("-dev CUDA0" in mm["cmd"] and mm["ttl"] == 0 and "GPU-de660e90" in str(mm["env"])
          and y["macros"]["server_mirai_s"] == "C:/x/llama-server.exe", "mirai-s: on the 5060 Ti, its engine")
    try:
        edit_config(new, macro, block)
        check(False, "a second deploy is refused")
    except ValueError:
        check(True, "a second deploy is refused")
    tb = edit_table(read(TABLE), 98304, False, 3072)
    ty = yaml.safe_load(tb)
    check(ty["models"]["mirai-s"]["window"]["ctx"] == 98304 and ty["models"]["mirai-s"]["window"]["main_cap"] == 95232
          and ty["models"]["mirai-s"]["vision"] is False and ty["models"]["bonsai"] == yaml.safe_load(read(TABLE))[
              "models"]["bonsai"], "the table: mirai-s's window and vision; bonsai's profile untouched")
    check(edit_table(tb, 98304, False, 3072) == tb, "the table edit is idempotent")
    bat = edit_bat(read("scripts/start-stack.bat"))
    check(bat.count('set "YAMADORI_TIER_MODELS=mcp\\tier_models.yaml"') == 1 and edit_bat(bat) == bat,
          "start-stack.bat: YAMADORI_TIER_MODELS once, idempotent")
    ps = edit_watchdog(read("scripts/watchdog.ps1"))
    check(ps.count("EnvTier = @{ YAMADORI_TIER_MODELS = 'mcp\\tier_models.yaml' }") == 3 and "$svc.EnvTier" in ps
          and edit_watchdog(ps) == ps, "watchdog.ps1: three services, applied, idempotent")
    eng = edit_engines(read("engines/manifest.yaml"))
    check(eng.count("config_refs: [mirai-s]") == 1, "engines manifest: config_refs")
    mod = edit_models(read("models/manifest.yaml"), False)
    check(mod.count("config_entries: [mirai-s]") == 1, "models manifest: the weights in service (no mmproj)")
    print(f"\n{'all passed' if not bad else f'{bad} FAILED'}")
    return 1 if bad else 0


# ------------------------------------------------------------------- main --
def _ps(script: str, timeout: int = 180) -> str:
    r = subprocess.run(["powershell", "-NoProfile", "-Command", script], capture_output=True, text=True,
                       timeout=timeout, check=False)
    return (r.stdout or "").strip()[-400:]


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--gate", help="bench/mirai_s_gate.py's output directory (gate.json)")
    ap.add_argument("--ctx", type=int, help="the window, instead of the gate's fit")
    ap.add_argument("--no-mtp", action="store_true")
    ap.add_argument("--mirai-mmproj", action="store_true", help="its own projector on the 5060 Ti")
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    os.chdir(ROOT)
    if a.selftest:
        return selftest()
    if not a.gate or not os.path.exists(os.path.join(a.gate, "gate.json")):
        print("refusing: --gate DIR with the Mirai S gate's gate.json is required (bench/mirai_s_gate.py)")
        return 2
    gate = json.load(open(os.path.join(a.gate, "gate.json"), encoding="utf-8"))
    values, why = values_from_gate(gate, {"ctx": a.ctx, "mtp": not a.no_mtp, "mmproj": a.mirai_mmproj})
    if not values["mmproj"] and '\n  "bonsai-vision":\n' not in read("config.yaml").replace("\r\n", "\n"):
        why.append("no bonsai-vision entry: vision off the main card needs layout v2 (bench/deploy_layout_v2.py) "
                   "first, or --mirai-mmproj")
    if why:
        print(json.dumps({"verdict": "REFUSED", "why": why}, indent=1))
        return 2
    import build_engine as be
    shipped = (be.load_yaml(be.MANIFEST)["engines"][ENGINE].get("shipped") or {}).get("path")
    if not shipped or not os.path.exists(shipped):
        print(f"refusing: {ENGINE} has no shipped binary")
        return 2
    new, old = planned(values, shipped)
    if a.preview:
        for f, t in new.items():
            sys.stdout.writelines(difflib.unified_diff(_lf(old[f])[0].splitlines(True), _lf(t)[0].splitlines(True),
                                                       f"a/{f}", f"b/{f}", n=1))
        print(json.dumps(values))
        return 0
    import engine_corruption as ec
    import verify_artifacts as va
    import deploy_layout_v2 as dl
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
        if (argv_after, env_after) != (argv_before, env_before):
            raise ValueError("bonsai's argv or env changed")
        bad = untouched(old["config.yaml"], read("config.yaml"))
        if bad:
            raise ValueError(f"entries this deploy must not touch changed: {bad}")
        import tier_models
        t = tier_models.load({"YAMADORI_TIER_MODELS": TABLE})
        if t.model_for("xhigh") != MODEL_ID or not t.window(MODEL_ID):
            raise ValueError("the table does not serve xhigh from mirai-s with its window")
    except Exception as e:                                           # noqa: BLE001
        rollback(f"{type(e).__name__}: {e}")
    problems, ok = be.verify_deploy()
    errs = [str(x) for x in va.verify("config.yaml") if x.severity == "error" and str(x) not in before]
    print(json.dumps({"engines": problems, "engines_ok": ok, "models_new_errors": errs, "backups": backups,
                      "values": values}, indent=1))
    if problems or errs:
        rollback("checks failed")
    if a.dry:
        rollback("dry run: checks passed, nothing restarted")
    rec: dict = {"watchdog_stop": _ps(dl.PS_WATCHDOG_STOP, 60)}
    _ps(dl.PS_RESTART)
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
    rec["watchdog_start"] = _ps(dl.PS_WATCHDOG_START, 60)
    rec["next"] = ("python scripts/deploy_check.py --key-file PATH (it walks the three models: "
                   "mcp/test_live_stack.py --only tier_models)")
    print(json.dumps(rec, indent=1))
    return 0 if all(up.values()) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
