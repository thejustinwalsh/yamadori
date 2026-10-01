"""Deploy the three-slot KV layout (operator, 2026-09-28; docs/ENGINES.md "KV rank") with the line PHASE B
measured (bench/kv_rank.py). In the style of the 746c8af6 deploy: every file it touches is backed up and put
back if a check fails before the restart; then ALL services restart through the Scheduled Task, and this waits
for health. scripts/deploy_check.py is run separately (a deploy is not good until it exits 0).

What changes:

  config.yaml          `bonsai`'s engine (the server_nudge macro) -> the 0041 build; in `bonsai`'s argv:
                       -c 2N + CHILD, --kv-vram-cells N (or none: --no-tier, -c N, the pool all VRAM),
                       -np 3, --kv-unified (an explicit -np turns the unified pool off; the tiered cache needs
                       one stream), --spec-draft-window 16384 (0036). Nothing else in the argv or env changes.
  engines/manifest.yaml  the 0041 build becomes `shipped`, the running one the first `previous`.
  scripts/start-stack.bat, scripts/watchdog.ps1
                       the proxy config values for the proxy, the tools API and the worker:
                       YAMADORI_MAIN_CAP = N (budget THE CAP LAYOUT; the served kv_vram_cells says the same)
                       and YAMADORI_CHILD_TOKENS = CHILD (the child's window, not taken from main); with
                       --no-tier, YAMADORI_HELPER_TOKENS = CHILD instead (budget's split: main = N - CHILD).
  mcp/fixtures/bonsai_props.json
                       re-recorded from the SERVED /props after the restart (n_ctx, total_slots,
                       kv_vram_cells), and checked against what was intended.
  config.yaml, the retrieval entries (operator, 2026-09-28)
                       `--cache-ram 0` on `embeddings` (Qwen3-Embedding-0.6B) and, until it was removed
                       2026-10-01 (docs/REMOVED.md), `reranker` (Qwen3-Reranker-0.6B). Observed by the operator: their llama-servers committed 10.8 /
                       6.8 GB private with a 0.06 GB working set (paged out; system commit 90.7 / 97.7 GB).
                       The suspected cause, READ FROM THE SOURCE of the binary they run
                       (C:/Users/jwals/llamacpp-prism-official, tools/server/server-context.cpp and
                       common/common.h): neither entry sets --cache-ram or --no-cache-idle-slots, so the
                       defaults apply -- cache_ram_mib 8192 (common.h:621) and cache_idle_slots true
                       (common.h:618) -- and on EVERY task start each idle slot is saved to the host-RAM
                       prompt cache ([TAG_IDLE_SLOT_CLEAR] loop, server-context.cpp ~2319; that path does not
                       check the task type, unlike the LRU path's "cache prompts only for completion tasks",
                       ~1547), up to 8 GiB. `--cache-ram 0` disables the cache and with it the idle-slot save
                       (server-context.cpp ~1339). CONFIRMED from the embedder's own log, captured before the
                       unload (C:/Users/jwals/octo/emb-log-20260928.txt: repeated "making room for prompt cache
                       entry, removing oldest entry" -- the cache full at its limit; the reranker's,
                       rr-log-20260928.txt beside it, has no cache line in its 100 KB window), so the edit is
                       applied UNCONDITIONALLY to both (coordinator, 2026-09-28); --no-retrieval-cache-ram opts
                       out. The retrieval record cites both files, the bytes before the unload, and the
                       committed bytes now and right after the restart (--commit re-records them after use).

    python bench/deploy_kv_rank.py --line N [--child 65536] [--engine-dir DIR] [--no-tier] [--dry]
    python bench/deploy_kv_rank.py --line N --preview     # the edits as diffs; no file is written
    python bench/deploy_kv_rank.py --commit               # the retrieval servers' committed bytes now

The edits are pure functions of the files' text (edit_config, edit_manifest, edit_bat, edit_watchdog), so
--preview shows them without touching the live files: llama-swap reloads a changed config.yaml, which a
--dry run would do twice.
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

NEW_DEFAULT = "C:/Users/jwals/engines/llama-bonsai2-ada-80d2c60d"
FILES = ["config.yaml", "engines/manifest.yaml", "scripts/start-stack.bat", "scripts/watchdog.ps1",
         "mcp/fixtures/bonsai_props.json"]
MARK = "kv layout (bench/deploy_kv_rank.py)"
WINDOW = "16384"


def read(f: str) -> str:
    return open(os.path.join(ROOT, f), encoding="utf-8", newline="").read()


def write(f: str, s: str) -> None:
    open(os.path.join(ROOT, f), "w", encoding="utf-8", newline="").write(s)


def _lf(s: str) -> tuple[str, str]:
    return s.replace("\r\n", "\n"), ("\r\n" if "\r\n" in s else "\n")


# ------------------------------------------------------------------ edits --
def edit_config(text: str, new_bin: str, n: int, child: int, no_tier: bool) -> tuple[str, str]:
    """(config.yaml with the layout, the binary it ran before)."""
    c, nl = _lf(text)
    ctx = n if no_tier else 2 * n + child
    m = re.search(r'^  server_nudge: "([^"]+)"$', c, re.M)
    if not m:
        raise ValueError("server_nudge macro not found")
    old_bin = m.group(1)
    c = c[:m.start()] + f'  server_nudge: "{new_bin}"' + c[m.end():]
    i0 = c.index('\n  "bonsai":\n')
    i1 = re.search(r'\n  "[^"]+":\n', c[i0 + 5:]).start() + i0 + 5
    blk = c[i0:i1]
    if blk.count("\n      -c ") != 1:
        raise ValueError("bonsai's -c line not found once")
    blk = re.sub(r"\n      -c \d+\n", f"\n      -c {ctx}\n", blk)
    kv = re.findall(r"\n      --kv-vram-cells \d+\n", blk)
    if len(kv) > 1:
        raise ValueError("more than one --kv-vram-cells in bonsai's block")
    tail = "\n      --spec-draft-n-max-tail 4\n"
    if no_tier:
        blk = re.sub(r"\n      --kv-vram-cells \d+\n", "\n", blk)
        anchor = tail
    elif kv:
        blk = re.sub(r"\n      --kv-vram-cells \d+\n", f"\n      --kv-vram-cells {n}\n", blk)
        anchor = f"\n      --kv-vram-cells {n}\n"
    else:
        if tail not in blk:
            raise ValueError("--spec-draft-n-max-tail 4 not in bonsai's block")
        blk = blk.replace(tail, tail + f"      --kv-vram-cells {n}\n", 1)
        anchor = f"\n      --kv-vram-cells {n}\n"
    add = []
    if "\n      -np " not in blk:
        add.append("      -np 3")
    if "\n      --kv-unified" not in blk:
        # an explicit -np turns llama-server's unified pool off (-np 3: three streams of -c/3 each); the
        # tiered cache (0028) and the moves (0040/0041) need ONE stream
        add.append("      --kv-unified")
    if "--spec-draft-window" not in blk:
        add.append(f"      --spec-draft-window {WINDOW}")
    if add:
        if anchor not in blk:
            raise ValueError(f"anchor {anchor!r} not in bonsai's block")
        pool = ("-c = N: the whole pool in VRAM, no tier" if no_tier
                else f"-c = 2N + the child ({child})")
        note = (f"      # 2026-09-28 (operator; {MARK}): 3 slots = two conversation slots + one child\n"
                f"      # slot; the VRAM line N = {n} measured by bench/kv_rank.py; {pool};\n"
                f"      # the MTP draft context sized for a {WINDOW}-token window per slot (0036)\n")
        blk = blk.replace(anchor, anchor + note + "\n".join(add) + "\n", 1)
    return (c[:i0] + blk + c[i1:]).replace("\n", nl), old_bin


def edit_manifest(text: str, old_bin: str, engine_dir: str, report: dict, patches: list[str],
                  summary: str) -> str:
    """engines/manifest.yaml with the new build `shipped` and the running one the first `previous`."""
    new_bin = f"{engine_dir}/src/build/bin/llama-server.exe"
    files = {f["file"]: f["new"] for f in report.get("files") or []}
    if "llama-server.exe" not in files or len(files) < 12:
        raise ValueError(f"build report files: {sorted(files)}")
    m, nl = _lf(text)
    j0 = m.index("  llama-bonsai2-ada:")
    j1 = m.index("  llama-bonsai2-base:")
    eb = m[j0:j1]
    s0 = eb.index("    shipped:\n")
    s1 = eb.index("    previous:\n")
    old_shipped = eb[s0 + len("    shipped:\n"):s1]
    if old_bin.replace("\\", "/") not in old_shipped:
        raise ValueError(f"the running binary {old_bin} is not the manifest's shipped one")
    item, first = [f"      # REPLACED {time.strftime('%Y-%m-%d')} by {os.path.basename(engine_dir)} "
                   "(0041, the three-slot KV layout; kept for rollback)\n"], True
    for ln in old_shipped.splitlines(keepends=True):
        if ln.startswith("      #"):
            item.append(ln)
        elif first and ln.startswith("      ") and not ln.startswith("       "):
            item.append("      - " + ln[6:])
            first = False
        else:
            item.append("  " + ln)
    ours = [p for p in patches if p[:4].isdigit() and int(p[:4]) >= 34]
    lines = ["    shipped:\n",
             "      # Built by scripts/build_engine.py from this entry (base + 0001-0033 + 0034-0041, no UI)\n",
             f"      # into {engine_dir}. 0041: KV rank (docs/ENGINES.md \"KV rank\").\n",
             f"      # DEPLOYED {time.strftime('%Y-%m-%d')} by bench/deploy_kv_rank.py: {summary}.\n",
             f"      path: {new_bin}\n",
             f"      build_tree: {engine_dir}/src/build\n",
             f"      built: {report.get('finished')}\n",
             "      built_with_patches: [\"0001-0033 (bonsai-ada-surgery)\", " + ", ".join(ours) + "]\n",
             "      files:\n"]
    for name in sorted(files, key=lambda k: (k != "llama-server.exe", k)):
        f = files[name]
        lines.append(f"        {name + ':':<22} {{sha256: {f['sha256']}, size: {f['size']}}}\n")
    eb = eb[:s0] + "".join(lines) + "    previous:\n" + "".join(item) + eb[s1 + len("    previous:\n"):]
    return (m[:j0] + eb + m[j1:]).replace("\n", nl)


def edit_bat(text: str, envs: dict) -> str:
    """start-stack.bat: the values set before the first Python service starts (tools API, worker, proxy)."""
    bat, nl = _lf(text)
    bat = re.sub(rf"REM {re.escape(MARK)}\n(set \"YAMADORI_[A-Z_]+=\d+\"\n)+", "", bat)
    anchor = 'start "" /B "%PY%" "%CD%\\mcp\\tools_api.py"'
    if bat.count(anchor) != 1:
        raise ValueError("start-stack.bat: the tools_api start line not found once")
    block = f"REM {MARK}\n" + "".join(f'set "{k}={v}"\n' for k, v in envs.items())
    return bat.replace(anchor, block + anchor, 1).replace("\n", nl)


def edit_watchdog(text: str, envs: dict) -> str:
    """watchdog.ps1: the same values for a watchdog restart of the proxy, the tools API and the worker (a
    restart inherits the watchdog's environment, not start-stack.bat's), as a second hashtable `Env2` that
    the restart applies after `Env`."""
    ps, nl = _lf(text)
    ps = re.sub(r"\n +# " + re.escape(MARK) + r"\n +Env2 = @\{[^}]*\}", "", ps)
    pairs = "; ".join(f"{k} = '{v}'" for k, v in envs.items())
    for match in ("Match = 'mcp[\\\\/]server\\.py'", "Match = 'tools_api\\.py'",
                  "Match = 'mcp[\\\\/]worker\\.py'"):
        if ps.count(match) != 1:
            raise ValueError(f"watchdog.ps1: {match} not found once")
        ps = ps.replace(match, match + f"\n       # {MARK}\n       Env2 = @{{ {pairs} }}", 1)
    if "$svc.Env2" not in ps:
        old = "    if ($svc.Env) {\n"
        if ps.count(old) != 1:
            raise ValueError("watchdog.ps1: the Env loop not found once")
        ps = ps.replace(old, old.replace("Env", "Env2")
                        + "        foreach ($k in $svc.Env2.Keys) {\n"
                        "            [Environment]::SetEnvironmentVariable($k, $svc.Env2[$k], 'Process')\n"
                        "        }\n    }\n" + old, 1)
    return ps.replace("\n", nl)


# `reranker` (Qwen3-Reranker-0.6B) was the second entry until its removal, 2026-10-01 (docs/REMOVED.md).
RETRIEVAL = {"embeddings": "Qwen3-Embedding-0.6B"}
CACHE_LINES = re.compile(r"prompt cache|cache size limit|cache state:|saving idle slot|making room for prompt "
                         r"cache|cache token limit", re.I)


def edit_retrieval(text: str) -> str:
    """config.yaml with `--cache-ram 0` in the embeddings entry (once; idempotent)."""
    c, nl = _lf(text)
    for entry, model in RETRIEVAL.items():
        m = re.search(r'\n  "' + re.escape(entry) + r'":\n', c)
        if not m:
            raise ValueError(f"config.yaml: entry {entry!r} not found")
        i0 = m.start()
        nxt = re.search(r'\n  "[^"]+":\n', c[m.end():])
        i1 = m.end() + nxt.start() if nxt else len(c)
        blk = c[i0:i1]
        if f"{model}-Q8_0.gguf" not in blk:
            raise ValueError(f"config.yaml: entry {entry!r} does not load {model}")
        if "--cache-ram" in blk:
            continue
        pm = re.search(r"\n      --pooling \w+\n", blk)
        if not pm:
            raise ValueError(f"config.yaml: entry {entry!r} has no --pooling line to anchor on")
        note = (f"      # {MARK}, operator 2026-09-28: no host-RAM prompt cache. With the defaults\n"
                "      # (--cache-ram 8192, --cache-idle-slots on) every task start saved each idle slot to\n"
                "      # it: 10.8 / 6.8 GB committed for these two, 0.06 GB working set. 0 also turns the\n"
                "      # idle-slot save off (server-context.cpp: it requires the cache).\n"
                "      --cache-ram 0\n")
        blk = blk[:pm.end()] + note + blk[pm.end():]
        c = c[:i0] + blk + c[i1:]
    return c.replace("\n", nl)


def retrieval_commit() -> dict:
    """Each retrieval llama-server's private (committed) and working-set bytes, and the system commit.
    Windows only; never raises."""
    ps = r"""
$out = @{}
foreach ($p in Get-CimInstance Win32_Process -Filter "Name='llama-server.exe'") {
  foreach ($m in @('Qwen3-Embedding-0.6B')) {
    if ($p.CommandLine -like "*$m*") {
      $g = Get-Process -Id $p.ProcessId -ErrorAction SilentlyContinue
      if ($g) { $out[$m] = @{ pid = $p.ProcessId; private_bytes = $g.PrivateMemorySize64;
                              working_set_bytes = $g.WorkingSet64;
                              cache_ram_flag = ($p.CommandLine -match '--cache-ram') } }
    }
  }
}
$os = Get-CimInstance Win32_OperatingSystem
$out['system'] = @{ commit_used_bytes = ($os.TotalVirtualMemorySize - $os.FreeVirtualMemory) * 1024;
                    commit_limit_bytes = $os.TotalVirtualMemorySize * 1024 }
$out | ConvertTo-Json -Depth 4 -Compress
"""
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True,
                           timeout=60)
        d = json.loads(r.stdout.strip() or "{}")
    except Exception as e:                                           # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"[:200]}
    d["at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    return d


# THE EVIDENCE (captured 2026-09-28 ~08:56 by the coordinator from GET :11434/logs/stream/<model>, before the
# two servers were unloaded; 100 KB windows each). The embedder's log carries repeated WRN lines
# "srv alloc: - making room for prompt cache entry, removing oldest entry (size = ...)", 26 of them, entries
# of 3.7-32.2 MiB: its prompt
# cache is full at its limit. The reranker's window shows slot LCP/LRU selection and no cache line; its
# 6.8 GB commit with the same defaults is the rest of the case. With the operator's observation (10.8 / 6.8 GB
# private, 0.06 GB working set) and this script's `--commit` reading at 08:53 (11.61 GB / 7.26 GB private,
# 0.07 GB working sets; system commit 103.6 of 104.9 GB), --cache-ram 0 is applied UNCONDITIONALLY to both
# (coordinator, 2026-09-28); --no-retrieval-cache-ram is the opt-out. The reranker's file and bytes stay as the
# evidence recorded then; the reranker itself was removed 2026-10-01 (docs/REMOVED.md).
EVIDENCE_FILES = {"embeddings": "C:/Users/jwals/octo/emb-log-20260928.txt",
                  "reranker": "C:/Users/jwals/octo/rr-log-20260928.txt"}
EVIDENCE_BEFORE_UNLOAD = {"at": "2026-09-28T08:53:37-0400",
                          "Qwen3-Embedding-0.6B": {"private_bytes": 11612381184, "working_set_bytes": 68472832},
                          "Qwen3-Reranker-0.6B": {"private_bytes": 7257206784, "working_set_bytes": 68194304},
                          "system": {"commit_used_bytes": 103605239808, "commit_limit_bytes": 104866324480}}


def retrieval_evidence() -> dict:
    """The record the deploy prints: the captured logs (each file's prompt-cache line count, read again
    here when the file exists), the bytes before the unload, and the committed bytes now. Never raises."""
    ev: dict = {"files": {}, "before_unload": EVIDENCE_BEFORE_UNLOAD, "commit_now": retrieval_commit()}
    for entry, path in EVIDENCE_FILES.items():
        rec: dict = {"path": path}
        try:
            text = open(path, encoding="utf-8", errors="replace").read()
            lines = [ln.strip()[:200] for ln in text.splitlines() if CACHE_LINES.search(ln)]
            rec.update(prompt_cache_lines=len(lines), example=lines[:2])
        except OSError as e:
            rec["error"] = f"{type(e).__name__}: {e}"[:160]
        ev["files"][entry] = rec
    return ev


def envs_for(n: int, child: int, no_tier: bool) -> dict:
    return ({"YAMADORI_HELPER_TOKENS": str(child)} if no_tier
            else {"YAMADORI_MAIN_CAP": str(n), "YAMADORI_CHILD_TOKENS": str(child)})


# ------------------------------------------------------------------- main --
def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--line", type=int, help="N, the VRAM line bench/kv_rank.py measured")
    ap.add_argument("--child", type=int, default=65536)
    ap.add_argument("--engine-dir", default=NEW_DEFAULT)
    ap.add_argument("--no-tier", action="store_true", help="no --kv-vram-cells: -c N, the whole pool in VRAM")
    ap.add_argument("--dry", action="store_true", help="make and check the edits, then put everything back")
    ap.add_argument("--preview", action="store_true", help="print the edits as diffs; write nothing")
    ap.add_argument("--no-retrieval-cache-ram", action="store_true",
                    help="leave the embeddings entry as it is")
    ap.add_argument("--commit", action="store_true",
                    help="only print the retrieval servers' committed bytes now (read-only)")
    a = ap.parse_args(argv)
    if a.commit:
        print(json.dumps(retrieval_commit(), indent=1))
        return 0
    if not a.line:
        print("refusing: --line N is required (the VRAM line bench/kv_rank.py measured)")
        return 2
    n, child = a.line, a.child
    if n % 256 or n <= 0:
        print(f"refusing: --line {n} is not a positive multiple of 256")
        return 2
    new_bin = f"{a.engine_dir}/src/build/bin/llama-server.exe"
    if not os.path.exists(new_bin):
        print(f"refusing: no binary at {new_bin}")
        return 2
    import build_engine as be
    ctx = n if a.no_tier else 2 * n + child
    envs = envs_for(n, child, a.no_tier)
    summary = f"-np 3 --kv-unified, -c {ctx}, {'no tier' if a.no_tier else f'--kv-vram-cells {n}'}, " \
              f"--spec-draft-window {WINDOW}"
    report = json.load(open(os.path.join(a.engine_dir, "build-report.json"), encoding="utf-8"))
    patches = [p["file"] for p in be.load_yaml(be.MANIFEST)["engines"]["llama-bonsai2-ada"]["patches"]]
    old = {f: read(f) for f in FILES}
    cfg, old_bin = edit_config(old["config.yaml"], new_bin, n, child, a.no_tier)
    new = {"config.yaml": cfg,
           "engines/manifest.yaml": edit_manifest(old["engines/manifest.yaml"], old_bin, a.engine_dir, report,
                                                  patches, summary),
           "scripts/start-stack.bat": edit_bat(old["scripts/start-stack.bat"], envs),
           "scripts/watchdog.ps1": edit_watchdog(old["scripts/watchdog.ps1"], envs)}
    if not a.no_retrieval_cache_ram:
        # unconditional (coordinator, 2026-09-28: THE EVIDENCE above)
        new["config.yaml"] = edit_retrieval(new["config.yaml"])
    if a.preview:
        for f, t in new.items():
            sys.stdout.writelines(difflib.unified_diff(_lf(old[f])[0].splitlines(True), _lf(t)[0].splitlines(True),
                                                       f"a/{f}", f"b/{f}", n=1))
        return 0

    os.chdir(ROOT)
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
    # THE RETRIEVAL SERVERS' PROMPT CACHE (THE EVIDENCE): --cache-ram 0 unless opted out
    retrieval: dict = {"evidence": retrieval_evidence(), "applied": not a.no_retrieval_cache_ram,
                       "why": ("--no-retrieval-cache-ram" if a.no_retrieval_cache_ram else
                               "unconditional (coordinator, 2026-09-28): the embedder's captured log shows its "
                               "prompt cache full at its limit; both run the same defaults")}
    print(json.dumps({"retrieval_before": retrieval}, indent=1))
    try:
        for f, t in new.items():
            write(f, t)
        argv_after, env_after = ec.production_command()
        if env_after != env_before:
            raise ValueError("bonsai env changed")
        if argv_after[0].replace("\\", "/") != new_bin:
            raise ValueError(argv_after[0])

        def val(av, flag):
            return av[av.index(flag) + 1] if flag in av else None
        want = {"-c": str(ctx), "-np": "3", "--spec-draft-window": WINDOW,
                "--kv-vram-cells": None if a.no_tier else str(n)}
        got = {k: val(argv_after, k) for k in want}
        if got != want:
            raise ValueError(f"argv: {got} != {want}")
        if "--kv-unified" not in argv_after:
            raise ValueError("argv: --kv-unified missing (an explicit -np turns the unified pool off)")

        def rest(av):
            out, skip = [], False
            for x in av[1:]:
                if skip:
                    skip = False
                    continue
                if x == "--kv-unified":
                    continue
                if x in want:
                    skip = True
                    continue
                out.append(x)
            return out
        if rest(argv_after) != rest(argv_before):
            raise ValueError("bonsai argv changed beyond -c, -np, --kv-unified, --spec-draft-window, --kv-vram-cells")
    except Exception as e:                                           # noqa: BLE001
        rollback(f"{type(e).__name__}: {e}")

    problems, ok = be.verify_deploy()
    errs = [str(x) for x in va.verify("config.yaml") if x.severity == "error" and str(x) not in before]
    print(json.dumps({"engines": problems, "engines_ok": ok, "models_new_errors": errs, "backups": backups,
                      "argv0": argv_after[0], "summary": summary, "env": envs}, indent=1))
    if problems or errs:
        rollback("checks failed")
    if a.dry:
        rollback("dry run: checks passed, nothing restarted")

    ps_restart = r"""
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
    rec: dict = {"health": up, "seconds": round(time.time() - t0)}
    # the fixture, re-recorded from the SERVED /props
    try:
        with urllib.request.urlopen("http://127.0.0.1:11434/upstream/bonsai/props", timeout=60) as r:
            live = json.load(r)
        served = {"n_ctx": (live.get("default_generation_settings") or {}).get("n_ctx"),
                  "total_slots": live.get("total_slots"), "kv_vram_cells": live.get("kv_vram_cells")}
        intended = {"n_ctx": ctx, "total_slots": 3, "kv_vram_cells": 0 if a.no_tier else n}
        rec.update(served=served, intended=intended, as_intended=served == intended)
        fx = json.loads(read("mcp/fixtures/bonsai_props.json"))
        fx.update(served, recorded=time.strftime("%Y-%m-%d"))
        fx["_what"] = re.sub(r"PLANNED LAYOUT, NOT YET SERVED[^.]*\.",
                             "RECORDED from the served /props by bench/deploy_kv_rank.py.", fx["_what"])
        write("mcp/fixtures/bonsai_props.json", json.dumps(fx, indent=2) + "\n")
    except Exception as e:                                           # noqa: BLE001
        rec["fixture"] = f"not re-recorded: {type(e).__name__}: {e}"
    # the retrieval servers' committed bytes: before (above) and right after the restart. Fresh processes
    # hold nothing yet, so the fair "after" is a later `--commit` once they have served the same work.
    retrieval["commit_after_restart"] = retrieval_commit()
    rec["retrieval"] = retrieval
    rec["next"] = ("python scripts/deploy_check.py --key-file PATH (a deploy is not good until it exits 0); "
                   "then python bench/deploy_kv_rank.py --commit after use, to compare with "
                   "retrieval.evidence.commit")
    print(json.dumps(rec, indent=1))
    return 0 if all(up.values()) and rec.get("as_intended") else 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
