#!/usr/bin/env python
"""Run every test suite in the repo and print one table. Standard library only.

    python scripts/run_tests.py             # every offline suite + ruff E9,F
    python scripts/run_tests.py --live      # + the opt-in suites that need the
                                            #   card or a live service
    python scripts/run_tests.py -v          # print every suite's full output
    python scripts/run_tests.py -k tiers    # only suites whose path contains it
    python scripts/run_tests.py --live --list   # show what would run, run nothing
    python scripts/run_tests.py --live --live-only --key-file PATH
                                            # THE DEPLOY GATE: only the live
                                            #   suites, key read from a file
    ... --maintenance                       # + the intrusive live tests (they
                                            #   restart the proxy)

WHAT IT RUNS

Every `mcp/test_*.py` and `bench/test_*.py` (and bench/domain, longctx,
swebench, octopus, sandbox, voxel and skills'), one process each, sequentially --
several suites build fixtures at fixed temp paths, and the live ones share one
GPU, so running them concurrently would produce the interference this stack
already lost a benchmark to. Then `ruff check mcp bench scripts --select=E9,F`,
which is a suite in its own right (PROTOCOL rule 14: it found two shipped
outages in under a second).

WHICH INTERPRETER

Every suite runs under the main environment (override with YAMADORI_PY).
The .venv-laya interpreter, for suites that imported torch, transformers or
the Laya training code, went with Laya's code on 2026-09-29.

LIVE SUITES ARE OPT-IN

`mcp/test_tools_live.py` needs the GPU and a second context on the same card;
without `--live` it is skipped and the skip is printed. With `--live`, every
suite that accepts `--live` gets it -- except the live arms retired with Laya
(RETIRED_LIVE, 2026-09-24, docs/E1.md), which run offline only.

OFFLINE SUITES NEVER REACH THE STACK

Every Python process an offline suite starts imports
scripts/offline_guard/sitecustomize.py (first on PYTHONPATH, switched on by
YAMADORI_OFFLINE_GUARD=1): a connect to a stack port -- 1234, 1235, 1237,
1238, 8888, 11434, 10001-10099 -- raises ConnectionRefusedError and is
logged, and the suite FAILS here with the frames that asked, whatever its
own fallback made of the outage. Found 2026-09-27: twelve suites read
llama-swap's /props (which loads `bonsai`), /running or the embedder. A suite
that needs the served model's state pins it (mcp/served_fixture.py);
mcp/test_live_stack.py `served` checks the pins against the live server.

OFFLINE SUITES NEVER WRITE THE LIVE STATE

Found 2026-09-27: an offline suite (mcp/test_domains.py, proxy.prepare with
no YAMADORI_JOBS_DB) wrote 21 skill fallback records into the live
index/jobs.sqlite3 between 2026-09-26 14:28 and 09-27 01:06, and the worker
learned from them. Two layers, because a live proxy and worker write the
same files while the suites run, so "the file changed" alone cannot say who
changed it:

  1. THE WRITE GUARD (exact attribution). The same sitecustomize refuses and
     logs every write the suite's own Python processes make under index/ or
     logs/: a writing open, a rename/remove/mkdir, and any write STATEMENT
     on a sqlite connection to a live database (a read-write connection
     may read; an authorizer refuses INSERT/UPDATE/DELETE/CREATE/DROP/
     ALTER and the file-changing PRAGMAs, and it never checkpoints on
     close; a suite that reads a live database should open it
     `file:...?mode=ro`). The audit hook runs in the suite's process, so a
     row in its log IS the suite -- never the stack running beside it. The
     suite fails and the frames that asked are printed.
  2. THE SNAPSHOT (a backstop for what the hook cannot see: a non-Python
     child such as node or a shell, or a process started with a cleared
     environment). Before and after each offline suite, every live-state
     file (LIVE_STATE_GLOBS) is stat'ed; one whose size or mtime moved is
     hashed and compared with its sha from the start of the run (files over
     SNAPSHOT_HASH_MAX are compared by size+mtime only). A change is then
     ATTRIBUTED by who else writes the file (LIVE_WRITERS):
       - files the running stack writes while it serves (the proxy's and the
         worker's databases, the seed, slot and power files, the trigger
         cache): reported as a note, never a failure -- the stack is the
         likelier writer, and a Python write by the suite is layer 1's;
       - files only a WORKER JOB writes (the skill library, the package
         registry history, the router labels): a failure unless the jobs
         database (read-only) shows a job active during the suite;
       - everything else (logs/deploy_check.jsonl, the embeddings, the pairs,
         the static indexes): a failure.
     `-shm` files are not compared: SQLite rewrites a WAL database's
     shared-memory index on ANY read, read-only included; it holds no rows.

The proxy on :1234 requires auth. An API key for it is read from
YAMADORI_TEST_KEY and handed to live suites in their environment under that
same name. It is never printed: it is masked in any output this script echoes,
and removed from the environment of offline suites, which have no use for it.

`--key-file PATH` reads the key from a file instead (the file is read, the
key goes into the live suites' environment, and it is never printed).

LIVE OUTPUT IS SHOWN AS IT HAPPENS

A live suite's output is streamed line by line (key masked): every check's
pass or FAIL with the evidence it printed. A live run takes an hour or more,
and a gate that shows nothing until the end is a gate nobody watches.

WHAT COUNTS AS A PASS

A suite passes when it exits 0 AND prints a count this script can read AND
that count is complete and non-zero. A suite that exits 0 without saying how
many checks ran is reported as a failure: a check that cannot fail is not a
check (PROTOCOL rules 3 and 16).

A 429 IS "NOT RUN". A live suite that was refused for load prints
"N tests NOT RUN (429)" and exits 3. That is never a pass and never a
failure: the suite is reported INCOMPLETE.

Exit code 1 if anything failed; 3 if nothing failed but something did not
run (a 429); 0 only when everything ran and passed. A deploy is good only
on 0 (scripts/deploy_check.py).
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


def _interpreter(env_name: str, default: str) -> str:
    p = os.environ.get(env_name) or default
    return p if os.path.exists(p) else sys.executable


MAIN_PY = _interpreter("YAMADORI_PY", _MAIN_DEFAULT)

# Needs the card: skipped unless --live.
LIVE_ONLY = {"mcp/test_tools_live.py", "mcp/test_live_stack.py"}
# Extra arguments a suite takes in --live mode. Anything else whose source
# accepts "--live" gets that flag.
LIVE_ARGS = {"mcp/test_tools_live.py": ["--live"],
             "mcp/test_live_stack.py": ["--live"]}
# Extra arguments under --maintenance: the intrusive live tests.
MAINTENANCE_ARGS = {"mcp/test_live_stack.py": ["--maintenance"]}
# LIVE ARMS RETIRED WITH LAYA (operator, 2026-09-24; docs/E1.md "Verdict:
# E1 alone. Retire Laya."). Their live arm called the Laya service on :1237
# (or served one on an alternate port), which no longer runs. They still run
# OFFLINE in every mode -- their cached numbers are the record docs/LAYA.md
# and docs/E1.md cite -- but --live never passes them --live/--serve, and
# --live-only skips them. bench/test_hint_collapse.py has no live arm since
# the reranker's removal (2026-10-01, docs/REMOVED.md): its Laya check had
# retired, and its reranker checks went with the reranker.
# EXPECTED FAILURES: checks that fail on purpose, each a documented, open
# finding. A suite whose ONLY failing checks are listed here passes the gate;
# they are named in the table and the summary every run, never hidden. A
# listed check that starts PASSING is reported too (the finding moved).
# Empty since 2026-10-01: its one entry, bench/test_hint_collapse.py's
# "scored AS A BATCH it does not" (docs/FINDINGS.md #20, the reranker's
# batched scores), went with the reranker.
EXPECTED_FAIL: dict[str, dict[str, str]] = {}
_FAIL_LINE = re.compile(r"^\s*FAIL\b:?\s+(.*?)\s*(?:\[.*|\(.*|<-.*)?$",
                        re.M)


def expected_only(name: str, output: str) -> tuple[bool, list[str], list[str]]:
    """(every failing check in `output` is an expected one, the expected
    checks that failed, the expected checks that did NOT fail)."""
    exp = EXPECTED_FAIL.get(name) or {}
    if not exp:
        return False, [], []
    failed = [m.group(1) for m in _FAIL_LINE.finditer(output)]
    hit = [k for k in exp if any(f.startswith(k) for f in failed)]
    other = [f for f in failed if not any(f.startswith(k) for k in exp)]
    missing = [k for k in exp if k not in hit]
    return bool(failed) and not other, hit, missing


RETIRED_LIVE = {
    "bench/test_guardrail.py": "Laya's live service check (:1237)",
}

# SUITES RETIRED WITH THEIR COMPONENT: skipped in every mode, named in the
# notes every run. Empty since 2026-09-29: the retired suites (mcp/test_clm.py,
# bench/test_laya_head.py, bench/test_laya_calibration.py) were deleted with
# their code (operator: "It should be in GitHub if we want to go back"; the
# way back is commit e360d37).
RETIRED_SUITES: dict[str, str] = {}

# SUITES THAT NEED A TOOL INSTALLED OUTSIDE THE STACK: skipped in every mode,
# with a note naming the install, until every path exists. The TypeSafe SDKs
# (mcp/test_jev_sdk.py) are an operator-approved download into their own
# venv and node_modules (tools/typesafe-sdk/install.py --run).
_TS = os.path.join("tools", "typesafe-sdk")
REQUIRES: dict[str, tuple[list[list[str]], str]] = {
    "mcp/test_jev_sdk.py": (
        [[os.path.join(_TS, "py", ".venv", "Scripts", "python.exe"),
          os.path.join(_TS, "py", ".venv", "bin", "python")],
         [os.path.join(_TS, "js", "node_modules", "@typesafe-ai", "sdk",
                       "package.json")]],
        "the TypeSafe SDKs: python tools/typesafe-sdk/install.py --run"),
}


def missing_requirement(name: str) -> str | None:
    """The install a suite still needs (REQUIRES), or None."""
    need = REQUIRES.get(name)
    if not need:
        return None
    groups, what = need
    for alternatives in groups:
        if not any(os.path.exists(os.path.join(ROOT, p)) for p in alternatives):
            return what
    return None

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


GUARD_DIR = os.path.join(ROOT, "scripts", "offline_guard")


def guard_hits(path: str) -> list[dict]:
    """The connects the offline guard refused, from its log at `path`
    (scripts/offline_guard/sitecustomize.py). Any row fails the suite."""
    import json
    rows = []
    try:
        with open(path, encoding="utf-8") as f:
            for ln in f:
                try:
                    rows.append(json.loads(ln))
                except ValueError:
                    rows.append({"port": "?", "host": "?", "stack": [ln[:200]]})
    except OSError:
        pass
    return rows


# ---------------------------------------------------------------------------
# LIVE STATE: the snapshot layer (the docstring, "OFFLINE SUITES NEVER WRITE
# THE LIVE STATE").
# ---------------------------------------------------------------------------
LIVE_STATE_GLOBS = (
    "index/*.sqlite3", "index/*.sqlite3-wal", "index/*.json", "index/*.jsonl",
    "index/*.npz", "index/*.key", "index/skills/**/*",
    "index/packages/registry_history.json", "index/packages/*.sqlite3",
    "index/accounts/**/*", "logs/deploy_check.jsonl")
# Who else writes a file (fnmatch on its repo-relative path, first match
# wins). "stack": the running proxy / worker / tools API write it while they
# serve -- a change is a note. "job": only a worker JOB writes it -- a change
# fails unless a job was active during the suite. Anything unlisted fails.
LIVE_WRITERS = (
    ("index/jobs.sqlite3*", "stack", "proxy + worker (jobs, skill records, "
                                     "deep decisions, selection counters)"),
    ("index/corpus.sqlite3*", "stack", "proxy (every request's turn)"),
    ("index/nebari.sqlite3*", "stack", "proxy (sessions, the ledger)"),
    ("index/rings.sqlite3*", "stack", "proxy (the work log)"),
    ("index/token_ledger.sqlite3*", "stack", "proxy (token accounting)"),
    ("index/concept_seed_last.json", "stack", "proxy (the seed it drew)"),
    ("index/slots_state.json", "stack", "proxy (slot pins)"),
    ("index/gpu_room.json", "stack", "proxy / tools API / worker (A4000)"),
    ("index/power_ledger.json", "stack", "proxy (power sampling)"),
    # mcp/max_mode.py _write_state: the proxy's max-mode lease state, for
    # the worker and tools API to read (gpu_room.state_dir()/max_mode.json),
    # rewritten whenever a lease changes -- i.e. while the stack serves.
    ("index/max_mode.json", "stack", "proxy (max-mode lease state for the "
                                     "worker and tools API)"),
    ("index/skill_triggers.npz", "stack", "proxy / worker (trigger vectors "
                                          "rebuilt when the armed set moves)"),
    ("index/accounts/*", "stack", "proxy (keys created on the dashboard)"),
    ("index/harness_kit.sqlite3*", "stack", "proxy (HARNESS TOOLS entries, "
                                            "written through the dashboard API)"),
    # mcp/stats_store.py: every generation's speed and the dashboard's
    # JJAVA / SOKUDO series, written off the response path while the stack
    # serves (2026-10-01: 22 suites were failed for the stack's own rows)
    ("index/stats.sqlite3*", "stack", "proxy (generation stats for the "
                                      "dashboard's JJAVA and SOKUDO pages)"),
    ("index/skills/*", "job", "worker skill jobs (arm, learn: labels)"),
    ("index/packages/registry_history.json", "job", "worker deps jobs"),
    ("index/packages/*.sqlite3", "job", "worker deps jobs (package index)"),
)
SNAPSHOT_HASH_MAX = 64 * 1024 * 1024


def _live_files() -> list[str]:
    out: set[str] = set()
    for g in LIVE_STATE_GLOBS:
        for p in glob.glob(os.path.join(ROOT, g), recursive=True):
            if os.path.isfile(p) and not p.endswith("-shm"):
                out.add(os.path.normpath(p))
    return sorted(out)


def _sha(path: str) -> str | None:
    import hashlib
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    except OSError:
        return None
    return h.hexdigest()


class LiveState:
    """Size, mtime and (for files up to SNAPSHOT_HASH_MAX) sha of every
    live-state file, carried from suite to suite: a suite is compared with
    the state right before it, so a change the stack made between suites is
    never charged to the next one."""

    def __init__(self):
        self.seen: dict[str, tuple] = {}
        for p in _live_files():
            self.seen[p] = self._entry(p, None)

    @staticmethod
    def _entry(p: str, old: tuple | None) -> tuple:
        try:
            st = os.stat(p)
        except OSError:
            return (None, None, None)
        size, mt = st.st_size, st.st_mtime_ns
        if old and old[0] == size and old[1] == mt:
            return old
        sha = _sha(p) if size <= SNAPSHOT_HASH_MAX else None
        return (size, mt, sha)

    def diff(self) -> list[tuple[str, str]]:
        """[(repo-relative path, what changed)] since the last call; the
        state is advanced."""
        now = set(_live_files()) | set(self.seen)
        changed = []
        for p in sorted(now):
            old = self.seen.get(p)
            new = self._entry(p, old)
            self.seen[p] = new
            if old is None:
                if new[0] is not None:
                    changed.append((rel(p), "created"))
                continue
            if old == new:
                continue
            if new[0] is None:
                changed.append((rel(p), "deleted"))
            elif old[2] and new[2]:
                if old[2] != new[2]:
                    changed.append((rel(p), f"content changed ({old[0]} -> "
                                            f"{new[0]} bytes)"))
            elif (old[0], old[1]) != (new[0], new[1]):
                changed.append((rel(p), f"size/mtime changed ({old[0]} -> "
                                        f"{new[0]} bytes; not hashed)"))
        return changed


def live_writer(path: str) -> tuple[str | None, str]:
    import fnmatch
    for pat, kind, who in LIVE_WRITERS:
        if fnmatch.fnmatch(path, pat):
            return kind, who
    return None, "nothing else writes it"


def jobs_active(t0: float, t1: float) -> list[str]:
    """Worker jobs active in [t0, t1], read from the live jobs database
    READ-ONLY; [] when it cannot be read (the change then fails)."""
    import sqlite3
    db = os.path.join(ROOT, "index", "jobs.sqlite3")
    try:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=5)
        try:
            rows = con.execute(
                "SELECT id, queue FROM jobs WHERE started IS NOT NULL AND "
                "started <= ? AND COALESCE(finished, heartbeat, ?) >= ?",
                (t1, t1, t0)).fetchall()
        finally:
            con.close()
    except Exception:                                            # noqa: BLE001
        return []
    return [f"{q} {i}" for i, q in rows]


def attribute(changes: list[tuple[str, str]], t0: float, t1: float,
              suite_writes: set[str]) -> tuple[list[str], list[str]]:
    """(failures, notes) for the files a suite's window changed."""
    fails, notes = [], []
    jobs = None
    for path, what in changes:
        if path in suite_writes:
            continue                     # layer 1 already named it
        kind, who = live_writer(path)
        if kind == "stack":
            notes.append(f"{path} {what} -- {who} writes it; not charged "
                         f"to the suite")
            continue
        if kind == "job":
            if jobs is None:
                jobs = jobs_active(t0, t1)
            if jobs:
                notes.append(f"{path} {what} -- a worker job was active "
                             f"({', '.join(jobs[:3])}); not charged")
                continue
            fails.append(f"{path} {what} (only a worker job writes it, and "
                         f"none was active)")
            continue
        fails.append(f"{path} {what} ({who})")
    return fails, notes


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


# A live suite's "not run": a 429 (the stack refused for load), a route
# not deployed yet, or no key -- none of them says the feature is broken.
_NOT_RUN = re.compile(r"(\d+) tests? NOT RUN \((?:429|not deployed|no key)")


def not_run(output: str) -> int:
    m = list(_NOT_RUN.finditer(output))
    return int(m[-1][1]) if m else 0


def run_streamed(cmd: list[str], env: dict, timeout: float,
                 mask: "Masker") -> tuple[int, str, float]:
    """run(), but every line is printed (masked) as it arrives."""
    import threading
    t0 = time.time()
    # A piped child block-buffers its stdout: without this, "streamed" output
    # arrives in one lump when the suite ends (found on the first gate run).
    env = dict(env, PYTHONUNBUFFERED="1")
    try:
        p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True,
                             encoding="utf-8", errors="replace")
    except OSError as e:
        return 127, f"[run_tests] could not start {cmd[0]}: {e}", time.time() - t0
    out: list[str] = []
    killed = threading.Event()

    def _kill():
        killed.set()
        p.kill()
    timer = threading.Timer(timeout, _kill)
    timer.start()
    try:
        for line in p.stdout:
            line = mask(line.rstrip("\n"))
            out.append(line)
            print("        " + line, flush=True)
        p.wait()
    finally:
        timer.cancel()
    text = "\n".join(out)
    if killed.is_set():
        return 124, text + f"\n[run_tests] TIMED OUT after {timeout:.0f}s", \
            time.time() - t0
    return p.returncode, text, time.time() - t0


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
    ap.add_argument("--key-file", default="",
                    help="read the proxy API key from this file (never printed)")
    ap.add_argument("--live-only", action="store_true",
                    help="with --live: only the suites that run live (the "
                         "deploy gate); no offline suites, no ruff")
    ap.add_argument("--maintenance", action="store_true",
                    help="with --live: also the intrusive live tests (they "
                         "restart the proxy)")
    args = ap.parse_args(argv)

    key = os.environ.get("YAMADORI_TEST_KEY") or None
    if args.key_file:
        try:
            with open(args.key_file, encoding="utf-8") as f:
                key = f.read().strip() or None
        except OSError as e:
            print(f"  cannot read --key-file: {e.strerror or e}")
            return 1
    mask = Masker(key)
    base_env = dict(os.environ)
    base_env["PYTHONIOENCODING"] = "utf-8"
    if key:
        base_env["YAMADORI_TEST_KEY"] = key
    offline_env = {k: v for k, v in base_env.items() if k != "YAMADORI_TEST_KEY"}
    # The A4000 coordinator (mcp/gpu_room.py) is OFF for offline suites: a
    # suite whose fake leaves a path pointed at the real llama-swap must never
    # be able to unload a real model. mcp/test_gpu_room.py turns it back on
    # against its own fake stack.
    offline_env["YAMADORI_GPU_ROOM"] = "0"
    # THE OFFLINE GUARD (scripts/offline_guard/sitecustomize.py, 2026-09-27):
    # every Python process an offline suite runs refuses a connect to a port
    # the stack serves on, and records it; the suite then FAILS, whatever its
    # fallback did (see guard_hits). Offline suites read the served template
    # through llama-swap's /props, which reloads `bonsai` (docs/ENGINES.md).
    offline_env["YAMADORI_OFFLINE_GUARD"] = "1"
    offline_env["PYTHONPATH"] = os.pathsep.join(
        p for p in (GUARD_DIR, offline_env.get("PYTHONPATH", "")) if p)

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
                    + glob.glob(os.path.join(ROOT, "bench", "swebench", "test_*.py"))
                    # bench/octopus: the run watcher's detectors (no GPU).
                    + glob.glob(os.path.join(ROOT, "bench", "octopus", "test_*.py"))
                    # bench/voxel: request, extraction, statuses, the check's
                    # measures and the compare page (fakes; no GPU, no docker).
                    + glob.glob(os.path.join(ROOT, "bench", "voxel", "test_*.py"))
                    # bench/sandbox: the shared sandbox network and the
                    # harness box (no GPU, no docker).
                    + glob.glob(os.path.join(ROOT, "bench", "sandbox", "test_*.py"))
                    # bench/skills: the daily-work selection gate (a copy
                    # of the live skill store, deterministic stages).
                    + glob.glob(os.path.join(ROOT, "bench", "skills", "test_*.py")))
    suites = [s for s in suites if args.k in rel(s)]

    print(f"  main interpreter  {MAIN_PY}")
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
    # The live-state snapshot (layer 2), taken once, advanced per suite.
    live_state = (LiveState() if not args.list and not args.live_only
                  else None)
    stack_notes = 0
    for path in suites:
        name = rel(path)
        live_only = name in LIVE_ONLY
        if live_only and not args.live:
            notes.append(f"skipped {name}: opt-in, needs the GPU "
                         f"(run with --live)")
            print(f"  skip  {name}  (opt-in, needs the GPU; pass --live)")
            continue
        need = missing_requirement(name)
        if need:
            notes.append(f"skipped {name}: not installed ({need})")
            print(f"  skip  {name}  (needs {need})")
            continue
        if name in RETIRED_SUITES:
            notes.append(f"skipped {name}: retired ({RETIRED_SUITES[name]})")
            print(f"  skip  {name}  (retired: {RETIRED_SUITES[name]})")
            continue
        py = MAIN_PY
        extra: list[str] = []
        is_live_run = False
        if args.live and name in RETIRED_LIVE:
            if args.live_only:
                print(f"  skip  {name}  (live arm retired with Laya: "
                      f"{RETIRED_LIVE[name]}; docs/E1.md)")
                continue
            notes.append(f"{name}: live arm retired with Laya "
                         f"({RETIRED_LIVE[name]}); ran offline")
        elif args.live:
            if name in LIVE_ARGS:
                extra = list(LIVE_ARGS[name])
            elif accepts_live(path):
                extra = ["--live"]
            is_live_run = bool(extra)
            if args.maintenance and name in MAINTENANCE_ARGS:
                extra += MAINTENANCE_ARGS[name]
        if args.live_only and not is_live_run:
            continue
        env = base_env if is_live_run else offline_env
        timeout = args.live_timeout if is_live_run else args.timeout
        print(f"  run   {name}" + (f" {' '.join(extra)}" if extra else "")
              + ("  [live env]" if is_live_run else ""),
              flush=True)
        if args.list:
            print(f"          {py} -X utf8 {name} {' '.join(extra)}".rstrip())
            continue
        cmd = [py, "-X", "utf8", path, *extra]
        hits: list[dict] = []
        state_fails: list[str] = []
        if is_live_run:
            rc, out, secs = run_streamed(cmd, env, timeout, mask)
        else:
            import tempfile
            fd, guard_log = tempfile.mkstemp(prefix="offline_guard_",
                                             suffix=".jsonl")
            os.close(fd)
            if live_state is not None:
                live_state.diff()       # what the stack wrote before now
            t0 = time.time()
            try:
                rc, out, secs = run(cmd, dict(
                    env, YAMADORI_OFFLINE_GUARD_LOG=guard_log), timeout)
                hits = guard_hits(guard_log)
                if live_state is not None:
                    state_fails, snotes = attribute(
                        live_state.diff(), t0, time.time(),
                        {h.get("path") for h in hits
                         if h.get("kind") == "write"})
                    stack_notes += len(snotes)
                    if args.verbose:
                        for n in snotes:
                            print(f"        live state (note): {n}")
            finally:
                try:
                    os.remove(guard_log)
                except OSError:
                    pass
        out = mask(out)
        c = counts(out)
        nr = not_run(out) if is_live_run else 0
        incomplete = False
        if c is None:
            ok, why = False, "printed no check count"
        elif c[1] == 0 and not nr:
            ok, why = False, "ran zero checks"
        elif rc == 3 and nr and c[0] == c[1]:
            # Nothing failed; something was refused with a 429.
            ok, incomplete, why = True, True, f"INCOMPLETE: {nr} not run (429 / not deployed)"
        elif rc != 0 and expected_only(name, out)[0]:
            hit = expected_only(name, out)[1]
            ok, why = True, ("EXPECTED failure(s): " + "; ".join(
                f"{k!r} -- {EXPECTED_FAIL[name][k]}" for k in hit))
            notes.append(f"{name}: expected failure(s), not blocking: "
                         + "; ".join(hit))
        elif rc != 0:
            ok, why = False, f"exit code {rc}"
        elif c[0] != c[1]:
            ok, why = False, "count incomplete despite exit 0"
        else:
            ok, why = True, ""
            missing = expected_only(name, out)[2]
            if missing and name in EXPECTED_FAIL and is_live_run:
                notes.append(f"{name}: an EXPECTED failure now passes -- "
                             f"re-read its finding: " + "; ".join(missing))
        conns = [h for h in hits if h.get("kind", "connect") == "connect"]
        writes = [h for h in hits if h.get("kind") == "write"]
        if conns:
            # Whatever the suite's own verdict: its fallback may have hidden
            # the attempt, and a result that depends on whether the stack is
            # up is not an offline result.
            ports = sorted({str(h.get("port")) for h in conns})
            ok, why = False, (f"reached the live stack ({len(conns)} "
                              f"connect(s) to port {', '.join(ports)} refused "
                              f"by the offline guard)")
        if writes:
            paths = sorted({str(h.get("path")) for h in writes})
            ok, why = False, ((why + "; " if not ok and conns else "")
                              + f"wrote the live state ({len(writes)} "
                              f"write(s) refused by the offline guard: "
                              f"{', '.join(paths[:4])})")
        if state_fails:
            ok, why = False, ((why + "; " if not ok and hits else "")
                              + f"changed the live state: "
                              f"{'; '.join(state_fails[:4])}")
        rows.append({"suite": name + (" " + " ".join(extra) if extra else ""),
                     "passed": c[0] if c else None, "total": c[1] if c else None,
                     "secs": secs, "ok": ok, "why": why,
                     "incomplete": incomplete})
        if is_live_run:
            continue                      # already shown line by line
        if args.verbose or not ok:
            tail = out if args.verbose else "\n".join(out.strip().splitlines()[-40:])
            print("\n".join("        " + ln for ln in tail.splitlines()))
        seen_at: set[tuple] = set()
        for h in hits:
            key_at = (h.get("host"), h.get("port"), h.get("path"),
                      tuple(h.get("stack") or ()))
            if key_at in seen_at:
                continue
            seen_at.add(key_at)
            target = (f"write {h.get('op')} {h.get('path')}"
                      if h.get("kind") == "write"
                      else f"{h.get('host')}:{h.get('port')}")
            print(f"        OFFLINE GUARD: {target} "
                  f"(pid {h.get('pid')}, {h.get('argv', '')[:80]})")
            for fr in (h.get("stack") or [])[-6:]:
                print(f"            {fr}")
        for f in state_fails:
            print(f"        LIVE STATE CHANGED: {f}")

    if args.list:
        return 0
    if stack_notes:
        notes.append(f"{stack_notes} change(s) to files the running stack "
                     f"writes happened during offline suites and were not "
                     f"charged to them (layer 1, the write guard, is what "
                     f"attributes a suite's own writes; -v lists them)")

    if not args.no_ruff and not args.live_only:
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
        flag = (f"   FAIL: {r['why']}" if not r["ok"] else
                f"   {r['why']}" if r.get("incomplete")
                or str(r.get("why", "")).startswith("EXPECTED") else "")
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
    incomplete = [r for r in rows if r.get("incomplete")]
    if failed:
        print(f"  {len(failed)} of {len(rows)} suites FAILED: "
              + ", ".join(r["suite"] for r in failed))
        return 1
    if incomplete:
        print(f"  INCOMPLETE: nothing failed, but {len(incomplete)} suite(s) "
              f"had tests refused with a 429 (not run): "
              + ", ".join(r["suite"] for r in incomplete)
              + ". Not a pass. Run them again on an idle stack.")
        return 3
    if not rows:
        print("  nothing ran")
        return 1
    print(f"  all {len(rows)} suites passed")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
