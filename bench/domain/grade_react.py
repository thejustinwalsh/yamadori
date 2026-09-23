#!/usr/bin/env python
"""The `react` grader kind: behavioural tests for React 19.2 answers.

Tasks live in tasks/react-*/ and are indexed in tasks_react.jsonl (NOT in
tasks.jsonl -- `grade.py --build-index` skips them). grade.grade() dispatches
kind `react` here, so the contract is the same as every other kind: stages
extract | compile | test | error, and it never raises.

PIPELINE, per grade, in one warm sandbox (_work/react, installed from the
tracked env/react/package.json + package-lock.json):

  compile  the answer becomes solution.tsx next to the task's test.tsx and the
           shared lib/react_helpers.tsx; `tsc --strict` (TypeScript 7.0.2,
           jsx react-jsx, lib dom) over all of them. An error located in
           solution.tsx is `compile`; one located only in test.tsx (a missing
           export, a wrong prop or return type) is `test`; one anywhere else
           is the grader's own bug -> `error`.
  test     vitest 5.0.1 + jsdom 30.1.1 + @testing-library/react 16.3.3 +
           @testing-library/user-event 14.6.7 against react/react-dom 19.2.8,
           JSON reporter. Any failed assertion, a test file that failed to
           load, or an unhandled error during the run is `test`. A run that
           reports zero tests is `error` (a check that cannot fail is not a
           check -- PROTOCOL rule 16), as is a run that leaves no report.

The versions match the stack's own web/package.json (react 19.2.8,
@types/react 19.2.18, @types/react-dom 19.2.7, vite 8.3.0, vitest 5.0.1).
React is pinned to 19.2.x deliberately: 19.3.0 is `latest` on npm but the
stack's dependencies cap at <19.3 (docs/HANDOFF.md).

    python bench/domain/grade_react.py --setup         # npm ci the sandbox
    python bench/domain/grade_react.py --build-index   # tasks/react-*/ -> jsonl
    python bench/domain/grade_react.py ID ANSWER_FILE  # grade one reply
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import grade as G  # noqa: E402

DOMAIN = "react"
TASKS_JSONL = os.path.join(HERE, "tasks_react.jsonl")
ENV_DIR = os.path.join(G.ENV_DIR, "react")
SANDBOX = os.path.join(G.WORK, "react")
NODE_MODULES = os.path.join(SANDBOX, "node_modules")
TSC = os.path.join(NODE_MODULES, "typescript", "bin", "tsc")
VITEST = os.path.join(NODE_MODULES, "vitest", "vitest.mjs")

PINS = {
    "react": "19.2.8", "react-dom": "19.2.8",
    "@types/react": "19.2.18", "@types/react-dom": "19.2.7",
    "typescript": "7.0.2", "vite": "8.3.0", "vitest": "5.0.1",
    "jsdom": "30.1.1", "@testing-library/react": "16.3.3",
    "@testing-library/dom": "10.4.2", "@testing-library/user-event": "14.6.7",
    "@types/node": "24.13.6",
}

T_TSC = 120
T_VITEST = 60

SUPPORT = ("react_helpers.tsx", "react_setup.ts")

TSCONFIG = {
    "compilerOptions": {
        "strict": True, "target": "es2022", "module": "esnext",
        "moduleResolution": "bundler", "jsx": "react-jsx",
        "lib": ["es2023", "dom", "dom.iterable"], "types": ["node"],
        "skipLibCheck": True, "noEmit": True, "isolatedModules": True,
        "esModuleInterop": True, "allowImportingTsExtensions": True,
    },
    "files": ["solution.tsx", "test.tsx", "helpers.tsx", "setup.ts"],
}

# Plain .mjs so vite does not have to bundle a TS config on every run.
VITEST_CONFIG = """\
export default {
  cacheDir: '.vite',
  test: {
    environment: 'jsdom',
    include: ['test.tsx'],
    setupFiles: ['./setup.ts'],
    globals: false,
    watch: false,
    testTimeout: 10000,
    hookTimeout: 10000,
    unstubGlobals: true,
    restoreMocks: true,
    reporters: ['json'],
    outputFile: 'result.json',
  },
};
"""


def ready() -> str | None:
    """None if the sandbox holds exactly the pinned packages, else why."""
    if not G.NODE:
        return "node not found (set DOMAIN_NODE)"
    for pkg, ver in PINS.items():
        got = G._read_json(os.path.join(NODE_MODULES, pkg, "package.json")
                           ).get("version")
        if got != ver:
            return (f"{pkg} in {SANDBOX} is {got!r}, need {ver} "
                    f"(run grade_react.py --setup)")
    for fn in SUPPORT:
        if not os.path.isfile(os.path.join(G.LIB_DIR, fn)):
            return f"lib/{fn} missing"
    return None


def setup(verbose: bool = True) -> int:
    os.makedirs(SANDBOX, exist_ok=True)
    for fn in ("package.json", "package-lock.json"):
        src = os.path.join(ENV_DIR, fn)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(SANDBOX, fn))
    if ready():
        cmd = "ci" if os.path.isfile(os.path.join(SANDBOX,
                                                  "package-lock.json")) else "install"
        r = G.run([G.NPM, cmd, "--no-audit", "--no-fund"], SANDBOX, 600)
        if verbose:
            print(r["out"][-800:], r["err"][-800:])
    why = ready()
    if verbose:
        print("react sandbox:", why or "ready")
    return 1 if why else 0


def _write(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def _failures(report: dict) -> list[str]:
    out = []
    for f in report.get("testResults") or []:
        for a in f.get("assertionResults") or []:
            if a.get("status") == "failed":
                msg = "\n".join(a.get("failureMessages") or [])
                # First line of each message is the assertion; drop the
                # stack frames that point into the test runner.
                keep = [l for l in msg.strip().splitlines()
                        if "node_modules" not in l and "<anonymous>" not in l]
                head = "\n".join(keep[:6])
                out.append(f"x {a.get('fullName') or a.get('title')}: {head}")
        if f.get("status") == "failed" and not f.get("assertionResults"):
            out.append("test file failed to load: "
                       + G._tail(f.get("message") or "", 1200))
    return out


def grade_react(task: dict, code: str, timings: dict) -> dict:
    g = task["grader"]
    why = ready()
    if why:
        return G._res(False, "error", why, error=True)
    box = os.path.join(SANDBOX, "tasks", G._safe(task["id"]))
    os.makedirs(box, exist_ok=True)
    for fn in ("solution.tsx", "test.tsx", "result.json"):
        try:
            os.remove(os.path.join(box, fn))
        except FileNotFoundError:
            pass
    _write(os.path.join(box, "solution.tsx"), code.rstrip() + "\n")
    shutil.copy2(G._p(g["test"]), os.path.join(box, "test.tsx"))
    shutil.copy2(os.path.join(G.LIB_DIR, "react_helpers.tsx"),
                 os.path.join(box, "helpers.tsx"))
    shutil.copy2(os.path.join(G.LIB_DIR, "react_setup.ts"),
                 os.path.join(box, "setup.ts"))
    cfg = json.loads(json.dumps(TSCONFIG))
    cfg["compilerOptions"].update(g.get("compilerOptions") or {})
    _write(os.path.join(box, "tsconfig.json"), json.dumps(cfg, indent=1))
    _write(os.path.join(box, "vitest.config.mjs"), VITEST_CONFIG)

    r = G.run([G.NODE, TSC, "-p", "tsconfig.json"], box, T_TSC)
    timings["tsc"] = round(r["secs"], 3)
    if r["spawn_failed"]:
        return G._res(False, "error", r["err"], error=True)
    if r["timeout"]:
        return G._res(False, "compile", "tsc timed out", timeout=True)
    if r["rc"] != 0:
        errs = [G._TSC_ERR.match(l) for l in (r["out"] + r["err"]).splitlines()]
        errs = [m for m in errs if m]
        if not errs:
            return G._res(False, "error", "tsc failed with no located error: "
                          + G._tail(r["out"] + r["err"]), error=True)
        base = lambda m: os.path.basename(m.group(1))            # noqa: E731
        in_sol = [m for m in errs if base(m) == "solution.tsx"]
        in_test = [m for m in errs if base(m) == "test.tsx"]
        other = [m for m in errs if m not in in_sol and m not in in_test]
        fmt = lambda ms: "\n".join(m.group(0) for m in ms[:12])  # noqa: E731
        if in_sol:
            return G._res(False, "compile", fmt(in_sol))
        if in_test:
            return G._res(False, "test", "the grader's usage does not "
                          "typecheck against the answer (missing export, "
                          "wrong props or return type):\n" + fmt(in_test))
        return G._res(False, "error", "tsc error outside the answer and the "
                      "grader:\n" + fmt(other), error=True)

    r = G.run([G.NODE, VITEST, "run"], box, T_VITEST,
              {"NO_COLOR": "1", "FORCE_COLOR": "0", "CI": "1"})
    timings["vitest"] = round(r["secs"], 3)
    if r["spawn_failed"]:
        return G._res(False, "error", r["err"], error=True)
    if r["timeout"]:
        return G._res(False, "test", f"vitest timed out after {T_VITEST}s "
                      "(an infinite render or update loop?)", timeout=True)
    rep = G._read_json(os.path.join(box, "result.json"))
    if not rep:
        return G._res(False, "error", "vitest left no JSON report (rc="
                      f"{r['rc']}):\n" + G._tail(r["out"] + "\n" + r["err"]),
                      error=True)
    total = rep.get("numTotalTests") or 0
    passed = rep.get("numPassedTests") or 0
    fails = _failures(rep)
    timings["tests"] = f"{passed}/{total}"
    if fails:
        return G._res(False, "test", f"{passed}/{total} tests pass\n"
                      + "\n".join(fails), tests_passed=passed,
                      tests_total=total)
    if total == 0:
        return G._res(False, "error", "vitest ran zero tests:\n"
                      + G._tail(r["out"] + r["err"]), error=True)
    if not rep.get("success") or r["rc"] != 0 or passed != total:
        # Every assertion passed but the run did not: an error thrown outside
        # a test (a leaked timer or listener firing after teardown).
        return G._res(False, "test", f"{passed}/{total} tests pass, but the run "
                      "failed (unhandled error outside a test):\n"
                      + G._tail(r["err"] or r["out"]), tests_passed=passed,
                      tests_total=total)
    return G._res(True, "test", f"typechecks; {passed}/{total} behavioural "
                  "tests pass", tests_passed=passed, tests_total=total)


# ----------------------------------------------------------------- index ---

def load_tasks(path: str = TASKS_JSONL) -> list[dict]:
    return G.load_tasks(path)


def load_task_dirs() -> list[dict]:
    rows = [G._read_json(p) for p in
            sorted(glob.glob(os.path.join(G.TASK_DIR, "react-*", "task.json")))]
    return sorted(rows, key=lambda t: t.get("id", ""))


def build_index() -> int:
    rows, problems = [], []
    need = ("id", "domain", "prompt", "grader", "reference", "contaminated",
            "needs_retrieval", "difficulty")
    for t in load_task_dirs():
        miss = [k for k in need if k not in t]
        if miss:
            problems.append(f"{t.get('id')}: missing {miss}")
            continue
        if t["domain"] != DOMAIN or t["grader"].get("kind") != "react":
            problems.append(f"{t['id']}: domain/kind must be react")
        for k in ("reference",):
            if not os.path.isfile(G._p(t[k])):
                problems.append(f"{t['id']}: {k} {t[k]} not found")
        if not os.path.isfile(G._p(t["grader"]["test"])):
            problems.append(f"{t['id']}: test {t['grader']['test']} not found")
        rows.append({k: t[k] for k in need} | {k: v for k, v in t.items()
                                                if k not in need})
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        problems.append("duplicate task ids")
    with open(TASKS_JSONL, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(rows)} tasks -> {TASKS_JSONL}")
    for pr in problems:
        print("  PROBLEM", pr)
    return 1 if problems else 0


def main() -> int:
    a = sys.argv[1:]
    if a[:1] == ["--setup"]:
        return setup()
    if a[:1] == ["--build-index"]:
        return build_index()
    if len(a) != 2:
        print(__doc__)
        return 2
    task = next((t for t in load_tasks() if t["id"] == a[0]), None)
    if task is None:
        print(f"no task {a[0]!r}")
        return 2
    with open(a[1], encoding="utf-8") as f:
        print(json.dumps(G.grade(task, f.read()), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
