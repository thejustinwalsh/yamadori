"""Deploy max mode's model (docs/FLASH-NEXT.md sections 4-5), AFTER the GPU gate (bench/flashnext_gate.py) passed.
In the style of bench/deploy_kv_rank.py: every file it touches is backed up and put back if a check fails before
the restart; then ALL services restart through the Scheduled Task and this waits for health. The deploy is not good
until `python scripts/deploy_check.py --key-file PATH` exits 0 (run separately), and a max-mode request has been
served live.

What changes:

  config.yaml            the macro `server_upstream_moe` (the llama-upstream-moe engine); the `flash-next` entry from
                         config.flash-next.fragment.yaml with the gate's values; the `primary` group gains
                         `flash-next` and `swap: true` (bonsai and flash-next swap each other on the 5060 Ti; nothing
                         else in the group or in `bonsai` changes). Preload stays bonsai only.
  scripts/start-stack.bat, scripts/watchdog.ps1
                         YAMADORI_MAX_MODEL=flash-next for the proxy, the tools API and the worker (mcp/max_mode.py
                         is off without it). YAMADORI_MAX_IDLE_S is NOT set: the operator has not given the value.
  engines/manifest.yaml  llama-upstream-moe's config_refs -> [flash-next].
  models/manifest.yaml   the flash-next files the entry loads: status in_service, config_entries [flash-next].

    python bench/deploy_flash_next.py --gate DIR --preview      # the edits as diffs; nothing is written
    python bench/deploy_flash_next.py --gate DIR --dry          # write, check, put everything back
    python bench/deploy_flash_next.py --gate DIR                # deploy (restarts the stack)

The values come from DIR/gate.json (fit.derived, and which arms passed) unless given: --n-cpu-moe, --cache-slots,
--ub, --np. A cache arm whose KL verdict is REPORT (outside Strata's yardstick) needs --accept-kl: the operator's
call, never automatic. The edits are pure functions of the files' text (edit_config, edit_bat, edit_watchdog,
edit_engines, edit_models), tested offline by --selftest.
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

ENGINE = "llama-upstream-moe"
FRAGMENT = "config.flash-next.fragment.yaml"
FILES = ["config.yaml", "scripts/start-stack.bat", "scripts/watchdog.ps1", "engines/manifest.yaml",
         "models/manifest.yaml"]
MARK = "max mode (bench/deploy_flash_next.py)"
MODEL_ID = "flash-next"
MODEL_FILES = ["flash-next-iq2xs-shard1", "flash-next-iq2xs-shard2-ngram", "flash-next-mmproj"]
ENVS = {"YAMADORI_MAX_MODEL": MODEL_ID}
LANE_TOKENS = 3072    # mcp/budget.py LANE_TOKENS (layout v2): the decider lane's cells
# the MTP arm's flags (bench/flashnext_gate.py `mtp`: Strata's --spec 4 --spec-min-p 0.5 as llama.cpp's; the draft
# layer's experts on the CPU like the trunk's)
MTP_ARGS = ("--spec-type draft-mtp --spec-draft-model ${models}/flash-next/mtp-Qwen3.8-Flash-Next-Q8_0-shared-embd.gguf "
            "-ngld 999 --spec-draft-n-cpu-moe 49 --spec-draft-n-max 3 --spec-draft-p-min 0.5")
QUANT_FILES = {"IQ2_XS": ["flash-next-iq2xs-shard1", "flash-next-iq2xs-shard2-ngram"],
               "Q2_0": ["flash-next-q2_0-shard1", "flash-next-q2_0-shard2-ngram"]}


def model_files(quant: str, mtp: bool, mmproj: bool = False) -> list[str]:
    return QUANT_FILES[quant] + (["flash-next-mmproj"] if mmproj else []) + (["flash-next-mtp-draft-q8_0"] if mtp else [])


def read(f: str) -> str:
    return open(os.path.join(ROOT, f), encoding="utf-8", newline="").read()


def write(f: str, s: str) -> None:
    open(os.path.join(ROOT, f), "w", encoding="utf-8", newline="").write(s)


def _lf(s: str) -> tuple[str, str]:
    return s.replace("\r\n", "\n"), ("\r\n" if "\r\n" in s else "\n")


# ------------------------------------------------------------------ values --
def values_from_gate(gate: dict, over: dict) -> tuple[dict, list[str]]:
    """The fragment's values and the reasons the gate does NOT allow a deploy (empty = allowed)."""
    why: list[str] = []
    arms = gate.get("arms") or {}
    if not (gate.get("kernels") or {}).get("PASS"):
        why.append("kernels: test-backend-ops did not pass (or did not run)")
    base = arms.get("base") or {}
    corr = base.get("corrupt") or {}
    if not corr or corr.get("error") or corr.get("guard"):
        why.append("base: the corruption checks did not run clean")
    if not (base.get("needles") or {}).get("PASS"):
        why.append("base: needles did not pass 15/15")

    def passed(arm: str, step: str) -> bool:
        r = (arms.get(arm) or {}).get(step) or {}
        return bool(r.get("PASS")) or r.get("verdict") == "PASS"
    derived = ((gate.get("fit") or {}).get("derived") or {})
    v = {"N_CPU_MOE": over.get("n_cpu_moe") or derived.get("n_cpu_moe_if_static"),
         # one conversation + the jjava lane (operator, 2026-09-29): -np 2, the native window + LANE_TOKENS
         "NP": over.get("np") or 2, "CTX": 262144 + (LANE_TOKENS if (over.get("np") or 2) == 2 else 0),
         "UB": over.get("ub") or 512,
         # the logical batch; the gate's flash arms ran -b 2048 -ub 512 (empty = -b equal to -ub, the first deploy's)
         "BATCH": over.get("batch") or over.get("ub") or 512,
         "EAGER": "EAGER" if passed("eager", "exact") and over.get("eager", True) else "LAZY",
         # pinned experts: the 2026-09-28 `pinned` arm's exact step, or --pin when the deployed arm itself ran pinned
         # (every 2026-09-29/30 flash arm sets LLAMA_PIN_EXPERTS=1)
         "PIN_EXPERTS": "1" if (passed("pinned", "exact") and over.get("pin", True)) or over.get("pin_arm") else "0",
         "PREFETCH_SLOTS": "3" if passed("prefetch", "exact") and over.get("prefetch", True) else "0",
         # one KV pool the slots share, as the gate ran it: without it upstream gives each slot -c/-np
         # (65,536 at -np 4; the gate's 128K needle was refused, 2026-09-28)
         "KV_UNIFIED": "--no-kv-unified" if over.get("no_kv_unified") else "--kv-unified",
         "PROFILE": "", "CACHE_POLICY": "", "CACHE_SLOTS": 0,
         # 0012: the attention KV in mapped host memory (the gate's kvmap arms)
         "KV_HOST_MAPPED": "1" if over.get("kv_host_mapped") else "0",
         # 0020: LLAMA_GRAPH_CACHE (the gate's gcache arm); 0 = off
         "GRAPH_CACHE": str(over.get("graph_cache") or 0),
         # 2026-09-29: the quant (the file), CPU threads, MTP and where the projector runs; each from an option,
         # the defaults are the first deploy's (IQ2_XS, llama.cpp's threads, no MTP, projector on the card)
         "QUANT": over.get("quant") or "IQ2_XS",
         "THREADS": over.get("threads") or "",
         "MTP": (MTP_ARGS if over.get("mtp") else ""),
         # layout v2 (operator, 2026-09-29): no projector on the main card by default
         "MMPROJ": ("--mmproj ${models}/flash-next/mmproj-Qwen3.8-Flash-Next-BF16.gguf"
                    if over.get("mmproj") == "card" else "")}
    cache = over.get("cache")            # "strata" | "lru" | None
    if cache:
        # the gate arm that measured this cache: 2026-09-30 cache-all (KV on the card, every layer's experts in RAM,
        # the adaptive cache over all 48; 0013-0016); before, the 2026-09-28 arms cache-strata / cache-lru
        arm = over.get("cache_arm") or ("cache-strata" if cache == "strata" else "cache-lru")
        kl = ((arms.get(arm) or {}).get("kl") or {}).get("verdict")
        if kl != "PASS" and not over.get("accept_kl"):
            why.append(f"{arm}: KL verdict {kl!r}; the operator decides (--accept-kl)")
        if not (arms.get(arm) or {}).get("needles", {}).get("PASS"):
            why.append(f"{arm}: needles did not pass")
        if over.get("cache_arm"):
            corr = (arms.get(arm) or {}).get("corrupt") or {}
            if not corr or corr.get("error") or corr.get("guard"):
                why.append(f"{arm}: the corruption checks did not run clean")
            if not ((arms.get(arm) or {}).get("contract") or {}).get("PASS"):
                why.append(f"{arm}: the llama-server contract did not pass")
        v["CACHE_SLOTS"] = over.get("cache_slots") or derived.get("cache_slots_per_cpu_layer") or 0
        if cache == "strata":
            v["PROFILE"] = "${models}/flash-next/expert-profile-strata-d551edf4.bin"
            v["CACHE_POLICY"] = "strata"
    if not v["N_CPU_MOE"]:
        why.append("no --n-cpu-moe: the gate's fit did not run and none was given")
    return v, why


# ------------------------------------------------------------------ edits --
def fragment(text: str, values: dict, engine_bin: str) -> tuple[str, str]:
    """(the macro line, the model block) from config.flash-next.fragment.yaml."""
    f, _ = _lf(text)
    vals = dict(values, ENGINE_BIN=engine_bin)
    for k, v in vals.items():
        f = f.replace("{{" + k + "}}", str(v))
    # an option left empty removes its line (and the comment line above it)
    lines, keep = f.split("\n"), []
    for ln in lines:
        if ln.strip() == "" and keep and keep[-1].lstrip().startswith("#") and ln.startswith("      "):
            while keep and keep[-1].lstrip().startswith("#") and keep[-1].startswith("      "):
                keep.pop()                       # every comment line that described the emptied option
            continue
        keep.append(ln)
    f = "\n".join(keep)
    left = re.findall(r"\{\{([A-Z_]+)\}\}", f)
    if left:
        raise ValueError(f"placeholders not filled: {left}")
    if engine_bin and not re.search(r"llama-upstream-(moe|flash)", engine_bin.replace("\\", "/")):
        # THE PLAIN ENGINE (--engine llama-upstream, 2026-09-28): the MoE build's own flags do not exist there, so
        # their lines go (with the comment line above each). The gate's base arm ran exactly this binary.
        lines = f.split("\n")
        keep: list[str] = []
        for ln in lines:
            if re.match(r"\s*--(moe-expert-cache|prefetch-experts-slots) ", ln):
                while keep and keep[-1].lstrip().startswith("#"):
                    keep.pop()                   # the flag's own comment lines
                continue
            keep.append(ln)
        f = "\n".join(keep)
    macro = re.search(r"^  server_upstream_moe: .*$", f, re.M).group(0)
    block = f[f.index('\n  "flash-next":\n') + 1:]
    block = re.sub(r" +$", "", block, flags=re.M).rstrip("\n") + "\n"
    block = block.replace("      -np 3 \n", "      -np 3\n")
    return macro, block


def _entry_end(c: str, i: int) -> int:
    """The end of the model entry whose name line starts at c[i] (LF text): the name line, then every line indented
    four spaces or more, or blank -- less the blank lines at its tail. A line at two spaces ("  # ...", the next
    entry's name) is not the entry's: a comment header after an entry (the CONTEXT TRIAL block, another deploy's
    "# tier models" label) belongs to what follows it and is never touched (2026-10-01: the region up to the next
    entry's name line swallowed ~50 such lines on a redeploy)."""
    end = c.find("\n", i)
    end = len(c) if end < 0 else end
    last = end
    while end < len(c):
        nxt = c.find("\n", end + 1)
        nxt = len(c) if nxt < 0 else nxt
        line = c[end + 1:nxt]
        if line.strip() == "":
            end = nxt
            continue
        if not line.startswith("    "):
            break
        end = last = nxt
    return last


def _strip(text: str) -> tuple[str, str, int | None, int | None, bool, bool, bool]:
    """(the LF text without the flash-next macro line and entry, its newline, where the macro line was, where the
    entry was, whether an entry was there, whether each of the two carried our marker line). Only the macro line
    (with our marker line right above it) and the entry (with our marker line and the blank line before it) are
    removed; edit_config puts the new ones back at the same places -- marked as they were -- so a redeploy changes
    nothing else byte for byte."""
    c, nl = _lf(text)
    had = '\n  "flash-next":\n' in c
    at_macro = at_entry = None
    macro_marked = entry_marked = True
    m = re.search(r"\n(  # " + re.escape(MARK) + r"\n)?  server_upstream_moe: [^\n]*", c)
    if m:
        c = c[:m.start()] + c[m.end():]
        at_macro = m.start()
        macro_marked = m.group(1) is not None
    if '\n  "flash-next":\n' in c:
        i0 = c.index('\n  "flash-next":\n')
        i1 = _entry_end(c, i0 + 1)
        entry = c[i0 + 1:i1]
        if "${server_upstream_moe}" not in entry:
            raise ValueError("config.yaml has a flash-next entry this script did not write: edit it by hand")
        # another deploy's comment lines inside our entry ("# tier models ...") are carried into the new entry
        CARRIED[:] = [ln for ln in entry.split("\n") if ln.startswith("    # tier models")]
        mark = "\n\n  # " + MARK
        entry_marked = c[:i0].endswith(mark)
        if entry_marked:
            i0 -= len(mark)
        c = c[:i0] + c[i1:]
        at_entry = i0
    return c, nl, at_macro, at_entry, had, macro_marked, entry_marked


def strip_flash_next(text: str) -> tuple[str, bool]:
    """config.yaml without a flash-next macro and entry this script wrote earlier (a REDEPLOY replaces them:
    a new engine, new values); (text, whether one was there). The group membership is kept."""
    c, nl, _, _, had, _, _ = _strip(text)
    return c.replace("\n", nl), had


# comment lines of another deploy inside an old flash-next entry, re-inserted under the new entry's name line
CARRIED: list[str] = []


def edit_config(text: str, macro: str, block: str) -> str:
    CARRIED[:] = []
    c, nl, at_macro, at_entry, _, macro_marked, entry_marked = _strip(text)
    if at_macro is None:
        m = re.search(r"^  server_nudge: .*$", c, re.M)
        if not m:
            raise ValueError("server_nudge macro not found")
        at_macro = m.end()
    ins = "\n" + (f"  # {MARK}\n" if macro_marked else "") + macro
    c = c[:at_macro] + ins + c[at_macro:]
    if at_entry is not None:
        i1 = at_entry + (len(ins) if at_entry >= at_macro else 0)   # where the old entry was
    else:
        i0 = c.index('\n  "bonsai":\n')                              # a first deploy: right after bonsai's entry
        i1 = _entry_end(c, i0 + 1)
    blk = block.rstrip("\n")
    if CARRIED:
        blk = blk.replace('  "flash-next":\n', '  "flash-next":\n' + "\n".join(CARRIED) + "\n", 1)
    c = c[:i1] + ("\n\n" + f"  # {MARK}\n" if entry_marked else "\n") + blk + c[i1:]
    g0 = c.index('\n        "primary":\n')
    g1 = c.index("\n\n", g0)
    grp = c[g0:g1]
    if '            - "flash-next"' in grp and "          swap: true" in grp:
        return c.replace("\n", nl)                          # a redeploy: the group already has both members
    if "          swap: false" not in grp or '            - "bonsai"' not in grp:
        raise ValueError("the primary group is not as expected")
    grp = grp.replace("          swap: false",
                      f"          # {MARK}: bonsai and flash-next swap each other on the 5060 Ti\n"
                      "          swap: true", 1)
    grp = grp.replace('            - "bonsai"', '            - "bonsai"\n            - "flash-next"', 1)
    return (c[:g0] + grp + c[g1:]).replace("\n", nl)


def edit_bat(text: str) -> str:
    bat, nl = _lf(text)
    bat = re.sub(rf"REM {re.escape(MARK)}\n(set \"YAMADORI_MAX_MODEL=[^\"]*\"\n)", "", bat)
    anchor = 'start "" /B "%PY%" "%CD%\\mcp\\tools_api.py"'
    if bat.count(anchor) != 1:
        raise ValueError("start-stack.bat: the tools_api start line not found once")
    block = f"REM {MARK}\n" + "".join(f'set "{k}={v}"\n' for k, v in ENVS.items())
    return bat.replace(anchor, block + anchor, 1).replace("\n", nl)


def edit_watchdog(text: str) -> str:
    """The same variable for a watchdog restart of the proxy, the tools API and the worker, as a hashtable `EnvMax`
    applied after `Env` (deploy_kv_rank.py uses `Env2`; the two never share a literal)."""
    ps, nl = _lf(text)
    ps = re.sub(r"\n +# " + re.escape(MARK) + r"\n +EnvMax = @\{[^}]*\}", "", ps)
    pairs = "; ".join(f"{k} = '{v}'" for k, v in ENVS.items())
    for match in ("Match = 'mcp[\\\\/]server\\.py'", "Match = 'tools_api\\.py'", "Match = 'mcp[\\\\/]worker\\.py'"):
        if ps.count(match) != 1:
            raise ValueError(f"watchdog.ps1: {match} not found once")
        ps = ps.replace(match, match + f"\n       # {MARK}\n       EnvMax = @{{ {pairs} }}", 1)
    if "$svc.EnvMax" not in ps:
        old = "    if ($svc.Env) {\n"
        if ps.count(old) != 1:
            raise ValueError("watchdog.ps1: the Env loop not found once")
        ps = ps.replace(old, "    if ($svc.EnvMax) {\n"
                        "        foreach ($k in $svc.EnvMax.Keys) {\n"
                        "            [Environment]::SetEnvironmentVariable($k, $svc.EnvMax[$k], 'Process')\n"
                        "        }\n    }\n" + old, 1)
    return ps.replace("\n", nl)


def _engine_block(t: str, engine: str) -> tuple[int, int]:
    i0 = t.index(f"\n  {engine}:\n")
    # the block ends at the next engine entry (or a section rule), never runs into the next engine's
    nxt = re.search(r"\n  [A-Za-z0-9_.-]+:\n|\n  # ----", t[i0 + 1:])
    return i0, (i0 + 1 + nxt.start() if nxt else len(t))


def edit_engines(text: str, engine: str = ENGINE) -> str:
    """config_refs [flash-next] on the engine deployed; an engine that served flash-next before gets [] (with a
    note: kept for rollback)."""
    t, nl = _lf(text)
    for other in re.findall(r"\n  ([A-Za-z0-9_.-]+):\n", t):
        if other == engine or f"\n  {other}:\n" not in t:
            continue
        i0, i1 = _engine_block(t, other)
        blk = t[i0:i1]
        new = re.sub(r"\n    config_refs: \[flash-next\][^\n]*",
                     f"\n    config_refs: []   # served flash-next until {engine} (bench/deploy_flash_next.py); "
                     "kept for rollback", blk, count=1)
        t = t[:i0] + new + t[i1:]
    i0, i1 = _engine_block(t, engine)
    blk = t[i0:i1]
    if "\n    config_refs: [flash-next]" in blk:
        return t.replace("\n", nl)
    new = re.sub(r"\n    config_refs: \[\][^\n]*", "\n    config_refs: [flash-next]", blk, count=1)
    if new == blk:
        raise ValueError(f"{engine}: config_refs [] not found")
    return (t[:i0] + new + t[i1:]).replace("\n", nl)


def edit_models(text: str, files: list[str] = MODEL_FILES) -> str:
    """The files the entry loads: in service with config_entries [flash-next]; a flash-next file the entry no
    longer loads (a quant switch) goes back to candidate with []."""
    t, nl = _lf(text)
    for aid in re.findall(r"\n  - id: (flash-next-[A-Za-z0-9_.-]+)\n", t):
        i0 = t.index(f"\n  - id: {aid}\n")
        nxt = t.find("\n  - id: ", i0 + 5)
        i1 = nxt if nxt >= 0 else len(t)
        blk = t[i0:i1]
        if aid in files:
            new = blk.replace("\n    status: candidate\n", "\n    status: in_service\n", 1)
            new = new.replace("\n    config_entries: []\n", "\n    config_entries: [flash-next]\n", 1)
            if "\n    config_entries: [flash-next]\n" not in new:
                raise ValueError(f"{aid}: neither a candidate with config_entries [] nor in service for flash-next")
        else:
            new = blk.replace("\n    config_entries: [flash-next]\n", "\n    config_entries: []\n", 1)
            if new != blk:
                new = new.replace("\n    status: in_service\n", "\n    status: candidate\n", 1)
        t = t[:i0] + new + t[i1:]
    missing = [f for f in files if f"\n  - id: {f}\n" not in t]
    if missing:
        raise ValueError(f"models/manifest.yaml has no entry for {missing}")
    return t.replace("\n", nl)


# -------------------------------------------------------------- selftest --
def selftest() -> int:
    bad = 0

    def check(ok, what):
        nonlocal bad
        print(("ok    " if ok else "FAIL  ") + what)
        bad += 0 if ok else 1
    gate = {"kernels": {"PASS": True},
            "fit": {"derived": {"n_cpu_moe_if_static": 40, "cache_slots_per_cpu_layer": 12}},
            "arms": {"base": {"corrupt": {"greedy": []}, "needles": {"PASS": True}},
                     "pinned": {"exact": {"PASS": True}}, "eager": {"exact": {"PASS": False}},
                     "cache-strata": {"kl": {"verdict": "REPORT (...)"}, "needles": {"PASS": True}}}}
    v, why = values_from_gate(gate, {})
    check(not why and v["N_CPU_MOE"] == 40 and v["PIN_EXPERTS"] == "1" and v["EAGER"] == "LAZY"
          and v["CACHE_SLOTS"] == 0, "the gate's fit and passed arms become the values")
    v2, why2 = values_from_gate(gate, {"cache": "strata"})
    check(any("KL verdict" in w for w in why2), "a cache arm outside the KL yardstick needs the operator")
    v3, why3 = values_from_gate(gate, {"cache": "strata", "accept_kl": True})
    check(not why3 and v3["CACHE_SLOTS"] == 12 and v3["CACHE_POLICY"] == "strata", "--accept-kl lets it through")
    _, why4 = values_from_gate({}, {})
    check(len(why4) >= 3, "no gate, no deploy")
    macro, block = fragment(read(FRAGMENT), v, "C:/x/llama-server.exe")
    check(macro == '  server_upstream_moe: "C:/x/llama-server.exe"' and '--n-cpu-moe 40' in block
          and "{{" not in block and block.startswith('  "flash-next":'), "the fragment fills")
    cfg, had = strip_flash_next(read("config.yaml"))
    if had:     # the live config holds the first deploy: undo the group edit too, to test a first deploy
        cfg = cfg.replace(f"          # {MARK}: bonsai and flash-next swap each other on the 5060 Ti\n", "")
        cfg = cfg.replace("          swap: true", "          swap: false", 1)
        # (wherever the member line sits: the tier models' deploy put mirai-s between them, 2026-10-01)
        cfg = cfg.replace('            - "flash-next"\n', '', 1)
    new = edit_config(cfg, macro, block)
    check('\n  "flash-next":' in new.replace("\r\n", "\n") and new.count('"flash-next"') >= 2,
          "config.yaml gains the entry and the group member")
    import yaml
    y = yaml.safe_load(new)
    grp = y["routing"]["router"]["settings"]["groups"]["primary"]
    check(grp["swap"] is True and {"bonsai", "flash-next"} <= set(grp["members"]) and grp["persistent"] is True,
          "the primary group: persistent, swap, both members", )
    check(y["models"]["bonsai"] == yaml.safe_load(cfg)["models"]["bonsai"], "bonsai's entry is untouched")
    fn = y["models"]["flash-next"]
    check("--mmproj" not in fn["cmd"] and "-dev CUDA0" in fn["cmd"] and fn["ttl"] == 0,
          "flash-next: no projector on the 5060 (layout v2: images go to bonsai-vision), the 5060")
    check(y["macros"]["server_upstream_moe"] == "C:/x/llama-server.exe", "the macro")
    check("-np 2 " in fn["cmd"] and "-c 265216 " in fn["cmd"],
          "one conversation + the jjava lane: -np 2, -c = the native window + LANE_TOKENS (operator, 2026-09-29)")
    check("--spec-type" not in fn["cmd"] and "-t " not in fn["cmd"] and "IQ2_XS-00001" in fn["cmd"]
          , "the defaults: IQ2_XS, no MTP, default threads")
    vq, _ = values_from_gate(gate, {"quant": "Q2_0", "mtp": True, "threads": "-t 16 -C 0xFFFF --cpu-strict 1",
                                    "mmproj": "card"})
    _, bq = fragment(read(FRAGMENT), vq, "C:/z/llama-upstream-flash-1/llama-server.exe")
    cq = yaml.safe_load("models:\n" + bq)["models"]["flash-next"]["cmd"]
    check("Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001" in cq and "--spec-type draft-mtp" in cq and "-C 0xFFFF" in cq
          and "--mmproj" in cq and "--moe-expert-cache" in cq,
          "the options fill: Q2_0, MTP, threads, projector on the CPU; a flash engine keeps the MoE flags")
    fq = yaml.safe_load("models:\n" + bq)["models"]["flash-next"]["env"]
    vk, _ = values_from_gate(gate, {"kv_host_mapped": True})
    _, bk = fragment(read(FRAGMENT), vk, "C:/z/llama-upstream-flash-1/llama-server.exe")
    check("LLAMA_KV_HOST_MAPPED=0" in fq and "LLAMA_KV_HOST_MAPPED=1" in
          yaml.safe_load("models:\n" + bk)["models"]["flash-next"]["env"], "0012's switch fills the env line")
    vg, _ = values_from_gate(gate, {"graph_cache": 8})
    _, bg = fragment(read(FRAGMENT), vg, "C:/z/llama-upstream-flash-1/llama-server.exe")
    check("LLAMA_GRAPH_CACHE=0" in fq and "LLAMA_GRAPH_CACHE=8" in
          yaml.safe_load("models:\n" + bg)["models"]["flash-next"]["env"], "0020's switch fills the env line")
    check(model_files("Q2_0", True, True) == ["flash-next-q2_0-shard1", "flash-next-q2_0-shard2-ngram", "flash-next-mmproj",
                                        "flash-next-mtp-draft-q8_0"], "a quant and MTP choose the model files")
    macro2, block2 = fragment(read(FRAGMENT), dict(v, N_CPU_MOE=38), "C:/y/llama-server.exe")
    again = edit_config(new, macro2, block2)
    y2 = yaml.safe_load(again)
    check(y2["macros"]["server_upstream_moe"] == "C:/y/llama-server.exe" and "--n-cpu-moe 38" in
          y2["models"]["flash-next"]["cmd"] and again.count('"flash-next":') == 1
          and y2["routing"]["router"]["settings"]["groups"]["primary"]["members"].count("flash-next") == 1
          and {k: w for k, w in y2["models"].items() if k != "flash-next"}
          == {k: w for k, w in y["models"].items() if k != "flash-next"},
          "a redeploy replaces the macro and the entry, nothing else")
    # A REDEPLOY REPLACES ONLY THE ENTRY (2026-10-01): today's layout -- a marked flash-next entry carrying another
    # deploy's comment line, then a comment header, a stray marker and the next entry's own label -- keeps every
    # line outside the entry and the macro's value byte for byte
    old_entry = ('  "flash-next":\n    # tier models (bench/deploy_tier_models.py): LOCKED\n    name: "old"\n'
                 '    env:\n      - "A=1"\n\n    cmd: |\n      ${server_upstream_moe}\n      --port ${PORT}\n    ttl: 0\n')
    after = ("\n  # =====================================================================\n"
             "  # CONTEXT TRIAL -- q4_0 K/V + K-cache mean-centering. NOT IN SERVICE.\n  #\n"
             "  # ttl 1800, not 0.\n  # =====================================================================\n\n"
             f"  # {MARK}\n\n  # tier models (bench/deploy_tier_models.py)\n"
             '  "mirai-s":\n    name: "m"\n    cmd: |\n      x\n    ttl: 0\n')
    fixture = ("macros:\n  server_nudge: \"C:/n.exe\"\n"
               f"  # {MARK}\n  server_upstream_moe: \"C:/old/llama-server.exe\"\n  models: \"C:/m\"\n"
               'models:\n  "bonsai":\n    name: "b"\n    cmd: |\n      y\n    ttl: 0\n\n'
               f"  # {MARK}\n" + old_entry + after +
               "routing:\n  router:\n    settings:\n      groups:\n        \"primary\":\n          swap: true\n"
               "          members:\n            - \"bonsai\"\n            - \"flash-next\"\n\n")
    got = edit_config(fixture, macro2, block2)
    want = (fixture.replace('"C:/old/llama-server.exe"', '"C:/y/llama-server.exe"')
            .replace(old_entry, block2.replace('  "flash-next":\n', '  "flash-next":\n'
                                               '    # tier models (bench/deploy_tier_models.py): LOCKED\n', 1)))
    check(got == want, "a redeploy replaces only the entry and the macro's value: the comment header after it, "
          "the stray marker and the next entry's label stay byte for byte")
    check(edit_config(got, macro2, block2) == got, "... and a second redeploy changes nothing")
    unmarked = fixture.replace(f"  # {MARK}\n" + old_entry, old_entry)
    check(edit_config(unmarked, macro2, block2) == want.replace(f"  # {MARK}\n" + block2.split("\n")[0], block2.split("\n")[0], 1),
          "an unmarked entry is replaced in place, unmarked")
    live = read("config.yaml")
    live_new = edit_config(live, macro2, block2)

    def outside(t: str) -> list[str]:
        c, _, _, _, _, _, _ = _strip(t)
        return c.split("\n")
    check(outside(live_new) == outside(live) and "CONTEXT TRIAL" in live_new
          and live_new.count(f"# {MARK}") == live.count(f"# {MARK}"),
          "today's config.yaml: every line outside the flash-next entry and macro survives a redeploy unchanged")
    bat = edit_bat(read("scripts/start-stack.bat"))
    check(bat.count('set "YAMADORI_MAX_MODEL=flash-next"') == 1 and edit_bat(bat) == bat,
          "start-stack.bat: YAMADORI_MAX_MODEL once, idempotent")
    ps = edit_watchdog(read("scripts/watchdog.ps1"))
    check(ps.count("EnvMax = @{ YAMADORI_MAX_MODEL = 'flash-next' }") == 3 and "$svc.EnvMax" in ps
          and edit_watchdog(ps) == ps, "watchdog.ps1: three services, applied, idempotent")
    eng = edit_engines(read("engines/manifest.yaml"))
    i0, i1 = _engine_block(_lf(eng)[0], ENGINE)
    check("config_refs: [flash-next]" in _lf(eng)[0][i0:i1] and _lf(eng)[0].count("config_refs: [flash-next]") == 1
          and edit_engines(eng) == eng, "engines manifest: config_refs on the deployed engine only, idempotent")
    mod = edit_models(read("models/manifest.yaml"), model_files("IQ2_XS", False))
    check(mod.count("config_entries: [flash-next]") == 2, "models manifest: the two shards in service, no projector")
    print(f"\n{'all passed' if not bad else f'{bad} FAILED'}")
    return 1 if bad else 0


# ------------------------------------------------------------------- main --
def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--gate", help="bench/flashnext_gate.py's output directory (gate.json)")
    ap.add_argument("--n-cpu-moe", type=int)
    ap.add_argument("--engine", default=ENGINE,
                    help="an engines/manifest.yaml entry with a shipped binary; llama-upstream = the plain engine "
                         "the gate's base arm ran (no MoE switches)")
    ap.add_argument("--cache", choices=["strata", "lru"])
    ap.add_argument("--cache-slots", type=int)
    ap.add_argument("--cache-arm", help="the gate arm that measured the cache (e.g. cache-all); its needles, corrupt,"
                                         " contract and KL decide")
    ap.add_argument("--pin", action="store_true", help="LLAMA_PIN_EXPERTS=1: the deployed arm ran pinned")
    ap.add_argument("--batch", type=int, help="-b (the logical batch); default = -ub")
    ap.add_argument("--accept-kl", action="store_true", help="the operator accepts a cache arm outside the yardstick")
    ap.add_argument("--no-kv-unified", action="store_true")
    ap.add_argument("--ub", type=int)
    ap.add_argument("--np", type=int)
    ap.add_argument("--quant", choices=sorted(QUANT_FILES), default="IQ2_XS")
    ap.add_argument("--threads", default="", help='e.g. "-t 16 -C 0xFFFF --cpu-strict 1" (the best thread arm of the gate)')
    ap.add_argument("--mtp", action="store_true", help="the MTP draft layer (the gate's mtp arm)")
    ap.add_argument("--graph-cache", type=int, default=0,
                    help="LLAMA_GRAPH_CACHE (0020): graph arenas kept per verify-batch size; 0 = off")
    ap.add_argument("--kv-host-mapped", action="store_true",
                    help="the attention KV in mapped host memory (0012; the gate's kvmap arms)")
    ap.add_argument("--mmproj", choices=["none", "card"], default="none",
                    help="none (layout v2, default): images go to bonsai-vision on the A4000; card: its own projector")
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    os.chdir(ROOT)
    if a.selftest:
        return selftest()
    if not a.gate or not os.path.exists(os.path.join(a.gate, "gate.json")):
        print("refusing: --gate DIR with the GPU gate's gate.json is required (bench/flashnext_gate.py)")
        return 2
    gate = json.load(open(os.path.join(a.gate, "gate.json"), encoding="utf-8"))
    values, why = values_from_gate(gate, {"n_cpu_moe": a.n_cpu_moe, "cache": a.cache, "cache_slots": a.cache_slots,
                                          "accept_kl": a.accept_kl, "no_kv_unified": a.no_kv_unified,
                                          "ub": a.ub, "np": a.np, "quant": a.quant, "threads": a.threads,
                                          "mtp": a.mtp, "mmproj": a.mmproj,
                                          "kv_host_mapped": a.kv_host_mapped, "cache_arm": a.cache_arm,
                                          "pin_arm": a.pin, "batch": a.batch, "graph_cache": a.graph_cache})
    if why:
        print(json.dumps({"verdict": "REFUSED: the gate does not allow a deploy", "why": why}, indent=1))
        return 2
    import build_engine as be
    shipped = (be.load_yaml(be.MANIFEST)["engines"][a.engine].get("shipped") or {}).get("path")
    if not shipped or not os.path.exists(shipped):
        print(f"refusing: {a.engine} has no shipped binary")
        return 2
    macro, block = fragment(read(FRAGMENT), values, shipped)
    old = {f: read(f) for f in FILES}
    new = {"config.yaml": edit_config(old["config.yaml"], macro, block),
           "scripts/start-stack.bat": edit_bat(old["scripts/start-stack.bat"]),
           "scripts/watchdog.ps1": edit_watchdog(old["scripts/watchdog.ps1"]),
           "engines/manifest.yaml": edit_engines(old["engines/manifest.yaml"], a.engine),
           "models/manifest.yaml": edit_models(old["models/manifest.yaml"], model_files(a.quant, a.mtp, a.mmproj == "card"))}
    if a.preview:
        for f, t in new.items():
            sys.stdout.writelines(difflib.unified_diff(_lf(old[f])[0].splitlines(True), _lf(t)[0].splitlines(True),
                                                       f"a/{f}", f"b/{f}", n=1))
        return 0
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
        if (argv_after, env_after) != (argv_before, env_before):
            raise ValueError("bonsai's argv or env changed")
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
    ps_restart = r"""
# every start-stack.bat cmd.exe first: a running batch file re-reads itself by byte offset, so a launcher alive from
# before this deploy edited start-stack.bat re-runs its tail and starts a proxy WITHOUT the new environment lines
# (2026-09-28 14:14, this deploy's first run; the same fix as bench/deploy_kv_rank.py)
Get-CimInstance Win32_Process -Filter "Name='cmd.exe'" | Where-Object { $_.CommandLine -like '*start-stack.bat*' } |
  ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
Get-Process llama-swap, llama-server -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
  Where-Object { $_.CommandLine -match 'mcp[\\/]server\.py|tools_api\.py|mcp[\\/]worker\.py|searx\.webapp' } |
  ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
Get-CimInstance Win32_Process -Filter "Name='wscript.exe'" -ErrorAction SilentlyContinue |
  Where-Object { $_.CommandLine -like '*start-stack-hidden*' } |
  ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
try { Stop-ScheduledTask -TaskName 'llama-stack' -ErrorAction SilentlyContinue } catch {}
Start-Sleep -Seconds 6
Start-ScheduledTask -TaskName 'llama-stack'
"""
    subprocess.run(["powershell", "-NoProfile", "-Command", ps_restart], check=False, timeout=180)
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
    print(json.dumps({"health": up, "seconds": round(time.time() - t0),
                      "next": "python scripts/deploy_check.py --key-file PATH; then one max-mode request live "
                              "(mcp/test_live_stack.py --only max_mode, to be written with the GPU window)"},
                     indent=1))
    return 0 if all(up.values()) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
