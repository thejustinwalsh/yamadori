#!/usr/bin/env python
"""Run every test suite in the repo and print one table. Standard library only.

    python scripts/run_tests.py             # every offline suite + ruff E9,F
    python scripts/run_tests.py --live      # + the opt-in suites that need the
                                            #   card or a live service
    python scripts/run_tests.py -v          # print every suite's full output
    python scripts/run_tests.py -k tiers    # only suites whose path contains it
    python scripts/run_tests.py --live --list   # show what would run, run nothing

WHAT IT RUNS

Every `mcp/test_*.py` and `bench/test_*.py`, one process each, sequentially --
several suites build fixtures at fixed temp paths, and the live ones share one
GPU, so running them concurrently would produce the interference this stack
already lost a benchmark to. Then `ruff check mcp bench scripts --select=E9,F`,
which is a suite in its own right (PROTOCOL rule 14: it found two shipped
outages in under a second).

WHICH INTERPRETER

A suite that imports torch, transformers or the Laya training code -- or
names `.venv-laya` in its own usage line -- runs under .venv-laya. Everything
else runs under the main environment. Override with YAMADORI_PY and
YAMADORI_LAYA_PY.

LIVE SUITES ARE OPT-IN

`mcp/test_tools_live.py` needs the GPU and a second context on the same card;
without `--live` it is skipped and the skip is printed. With `--live`, every
suite that accepts `--live` gets it, and `bench/test_laya_head.py` gets
`--serve` (an alternate-port Laya service, never 1237).

The proxy on :1234 requires auth. An API key for it is read from
YAMADORI_TEST_KEY and handed to live suites in their environment under that
same name. It is never printed: it is masked in any output this script echoes,
and removed from the environment of offline suites, which have no use for it.

WHAT COUNTS AS A PASS

A suite passes when it exits 0 AND prints a count this script can read AND
that count is complete and non-zero. A suite that exits 0 without saying how
many checks ran is reported as a failure: a check that cannot fail is not a
check (PROTOCOL rules 3 and 16).

Exit code 1 if anything failed, 0 otherwise.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

_MAIN_DEFAULT = r"C:/Users/jwals/textgen/installer_files/env/python.exe"
_LAYA_DEFAULT = os.path.join(ROOT, ".venv-laya", "Scripts", "python.exe")


def _interpreter(env_name: str, default: str) -> str:
    p = os.environ.get(env_name) or default
    return p if os.path.exists(p) else sys.executable


MAIN_PY = _interpreter("YAMADORI_PY", _MAIN_DEFAULT)
LAYA_PY = _interpreter("YAMADORI_LAYA_PY", _LAYA_DEFAULT)

# Needs the card: skipped unless --live.
LIVE_ONLY = {"mcp/test_tools_live.py", "mcp/test_live_stack.py"}
# Extra arguments a suite takes in --live mode. Anything else whose source
# accepts "--live" gets that flag.
LIVE_ARGS = {"mcp/test_tools_live.py": ["--live"],
             "mcp/test_live_stack.py": ["--live"],
             "bench/test_laya_head.py": ["--serve"]}

_LAYA_IMPORT = re.compile(
    r"^\s*(?:import|from)\s+(?:torch|transformers|laya_head|train_laya)\b", re.M)

_COUNTS = (
    (re.compile(r"(\d+)/(\d+) (?:live )?checks passed"),
     lambda m: (int(m[1]), int(m[2]))),
    (re.compile(r"(\d+) passed, (\d+) failed"),
     lambda m: (int(m[1]), int(m[1]) + int(m[2]))),
    (re.compile(r"(\d+) FAILED, (\d+) passed"),
     lambda m: (int(m[2]), int(m[1]) + int(m[2]))),
    (re.compile(r"all (\d+) checks passed"),
     lambda m: (int(m[1]), int(m[1]))),
)


def rel(p: str) -> str:
    return os.path.relpath(p, ROOT).replace(os.sep, "/")


def needs_laya(path: str) -> bool:
    with open(path, encoding="utf-8", errors="replace") as f:
        src = f.read()
    return bool(_LAYA_IMPORT.search(src)) or ".venv-laya" in src


def accepts_live(path: str) -> bool:
    with open(path, encoding="utf-8", errors="replace") as f:
        return '"--live"' in f.read()


def counts(output: str) -> tuple[int, int] | None:
    """The LAST count line any known harness prints, or None."""
    best = None
    for pat, conv in _COUNTS:
        for m in pat.finditer(output):
            if best is None or m.start() > best[0]:
                best = (m.start(), conv(m))
    return best[1] if best else None


class Masker:
    def __init__(self, secret: str | None):
        self.secret = secret or None

    def __call__(self, text: str) -> str:
        return text.replace(self.secret, "***") if self.secret else text


def run(cmd: list[str], env: dict, timeout: float) -> tuple[int, str, float]:
    t0 = time.time()
    try:
        r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True,
                           text=True, encoding="utf-8", errors="replace",
                           timeout=timeout)
        return r.returncode, (r.stdout or "") + (r.stderr or ""), time.time() - t0
    except subprocess.TimeoutExpired as e:
        out = e.stdout or ""
        if isinstance(out, bytes):
            out = out.decode("utf-8", "replace")
        return 124, out + f"\n[run_tests] TIMED OUT after {timeout:.0f}s", time.time() - t0
    except OSError as e:
        return 127, f"[run_tests] could not start {cmd[0]}: {e}", time.time() - t0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--live", action="store_true",
                    help="also run the opt-in suites that need the GPU or a live service")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="print every suite's full output")
    ap.add_argument("-k", default="", help="only suites whose path contains this")
    ap.add_argument("--timeout", type=float, default=900.0,
                    help="seconds per offline suite (default 900)")
    ap.add_argument("--live-timeout", type=float, default=7200.0,
                    help="seconds per live suite (default 7200)")
    ap.add_argument("--no-ruff", action="store_true", help="skip the ruff suite")
    ap.add_argument("--list", action="store_true",
                    help="print the command for each suite and run nothing")
    args = ap.parse_args(argv)

    key = os.environ.get("YAMADORI_TEST_KEY") or None
    mask = Masker(key)
    base_env = dict(os.environ)
    base_env["PYTHONIOENCODING"] = "utf-8"
    offline_env = {k: v for k, v in base_env.items() if k != "YAMADORI_TEST_KEY"}

    # bench/domain holds the benchmark's own graders and runner. Their proofs
    # (every reference passes, every wrong answer fails at its stage) are what
    # make a benchmark row mean anything, so they run with everything else.
    suites = sorted(glob.glob(os.path.join(ROOT, "mcp", "test_*.py"))
                    + glob.glob(os.path.join(ROOT, "bench", "test_*.py"))
                    + glob.glob(os.path.join(ROOT, "bench", "domain", "test_*.py"))
                    # bench/longctx: haystack, grader, resume and usable-context
                    # proofs for the long-context benchmark (no GPU).
                    + glob.glob(os.path.join(ROOT, "bench", "longctx", "test_*.py"))
                    # bench/swebench: the SWE-bench results parser (no GPU).
                    + glob.glob(os.path.join(ROOT, "bench", "swebench", "test_*.py")))
    suites = [s for s in suites if args.k in rel(s)]

    print(f"  main interpreter  {MAIN_PY}")
    print(f"  laya interpreter  {LAYA_PY}")
    if args.live:
        print("  mode              LIVE -- opt-in suites run against the card "
              "and live services")
        print("  YAMADORI_TEST_KEY " + ("set (passed to live suites, never printed)"
                                        if key else
                                        "NOT SET -- anything that calls the proxy "
                                        "on :1234 will be refused"))
    print()

    rows: list[dict] = []
    notes: list[str] = []
    for path in suites:
        name = rel(path)
        live_only = name in LIVE_ONLY
        if live_only and not args.live:
            notes.append(f"skipped {name}: opt-in, needs the GPU "
                         f"(run with --live)")
            print(f"  skip  {name}  (opt-in, needs the GPU; pass --live)")
            continue
        py = LAYA_PY if needs_laya(path) else MAIN_PY
        extra: list[str] = []
        is_live_run = False
        if args.live:
            if name in LIVE_ARGS:
                extra = list(LIVE_ARGS[name])
            elif accepts_live(path):
                extra = ["--live"]
            is_live_run = bool(extra)
        env = base_env if is_live_run else offline_env
        timeout = args.live_timeout if is_live_run else args.timeout
        print(f"  run   {name}" + (f" {' '.join(extra)}" if extra else "")
              + ("  [laya venv]" if py == LAYA_PY and py != MAIN_PY else "")
              + ("  [live env]" if is_live_run else ""),
              flush=True)
        if args.list:
            print(f"          {py} -X utf8 {name} {' '.join(extra)}".rstrip())
            continue
        rc, out, secs = run([py, "-X", "utf8", path, *extra], env, timeout)
        out = mask(out)
        c = counts(out)
        if c is None:
            ok, why = False, "printed no check count"
        elif c[1] == 0:
            ok, why = False, "ran zero checks"
        elif rc != 0:
            ok, why = False, f"exit code {rc}"
        elif c[0] != c[1]:
            ok, why = False, "count incomplete despite exit 0"
        else:
            ok, why = True, ""
        rows.append({"suite": name + (" " + " ".join(extra) if extra else ""),
                     "passed": c[0] if c else None, "total": c[1] if c else None,
                     "secs": secs, "ok": ok, "why": why})
        if args.verbose or not ok:
            tail = out if args.verbose else "\n".join(out.strip().splitlines()[-40:])
            print("\n".join("        " + ln for ln in tail.splitlines()))

    if args.list:
        return 0

    if not args.no_ruff:
        cmd = [MAIN_PY, "-m", "ruff", "check", "mcp", "bench", "scripts",
               "--select=E9,F", "--output-format=concise"]
        print("  run   ruff check mcp bench scripts --select=E9,F", flush=True)
        rc, out, secs = run(cmd, offline_env, args.timeout)
        if rc == 127 or "No module named ruff" in out:
            rc, out, secs = run(["ruff", *cmd[3:]], offline_env, args.timeout)
        out = mask(out)
        findings = len([ln for ln in out.splitlines()
                        if re.match(r"^\S.*:\d+:\d+: [A-Z]+\d+", ln)])
        ok = rc == 0
        rows.append({"suite": "ruff E9,F", "passed": 1 if ok else 0, "total": 1,
                     "secs": secs, "ok": ok,
                     "why": "" if ok else (f"{findings} findings" if findings
                                           else f"exit code {rc}")})
        if args.verbose or not ok:
            print("\n".join("        " + ln for ln in out.strip().splitlines()[-60:]))

    # ------------------------------------------------------------------ table
    w = max([len(r["suite"]) for r in rows] + [5])
    print()
    print(f"  {'suite':<{w}}  {'passed':>7}  {'total':>7}  {'seconds':>8}")
    print(f"  {'-' * w}  {'-' * 7}  {'-' * 7}  {'-' * 8}")
    for r in rows:
        p = "?" if r["passed"] is None else str(r["passed"])
        t = "?" if r["total"] is None else str(r["total"])
        flag = "" if r["ok"] else f"   FAIL: {r['why']}"
        print(f"  {r['suite']:<{w}}  {p:>7}  {t:>7}  {r['secs']:>8.1f}{flag}")
    gp = sum(r["passed"] or 0 for r in rows)
    gt = sum(r["total"] or 0 for r in rows)
    gs = sum(r["secs"] for r in rows)
    failed = [r for r in rows if not r["ok"]]
    print(f"  {'-' * w}  {'-' * 7}  {'-' * 7}  {'-' * 8}")
    print(f"  {'TOTAL':<{w}}  {gp:>7}  {gt:>7}  {gs:>8.1f}")
    print()
    for n in notes:
        print(f"  note: {n}")
    if failed:
        print(f"  {len(failed)} of {len(rows)} suites FAILED: "
              + ", ".join(r["suite"] for r in failed))
        return 1
    print(f"  all {len(rows)} suites passed")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
