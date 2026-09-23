#!/usr/bin/env python
"""Proof that the `react` grader is sound, before any model is scored by it.

Sibling of test_grade.py, for tasks_react.jsonl (tasks/react-*/). Every task
is run through `grade.grade` exactly as a model reply would be:

  (a) reference.tsx in a ```tsx fence       -> must PASS
  (b) every alt_*.tsx (a correct answer
      written a different way)              -> must PASS  (no false negatives)
  (c) every wrong_<stage>[_why].tsx         -> must FAIL at <stage>
                                               (compile | test), never error
  (d) prose with no code block              -> must FAIL at extract
  (e) the reference in a ```python fence    -> must FAIL at extract

Each task must carry at least one alt and at least two wrong_test answers.
100% of references must pass; one failing fails the run. None of the above
may come back as stage `error` -- that would be the grader, not the answer.

It also proves the ground is alive (PROTOCOL rule 1): every pinned package
in the sandbox is the pinned version, react at runtime reports 19.2.x, tsc
answers and is 7.0.2.

    python bench/domain/test_grade_react.py              # everything
    python bench/domain/test_grade_react.py --only react-0   # id prefix
    python bench/domain/test_grade_react.py --dirs       # read task dirs
    python bench/domain/test_grade_react.py --jobs 4     # parallel; timings
                                                         # are then contended
"""
from __future__ import annotations

import glob
import json
import os
import re
import statistics
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import grade  # noqa: E402
import grade_react  # noqa: E402

_results: list[tuple[bool, str, str]] = []
TIMES: list[dict] = []
ONLY: str | None = None
DIRS = False
JOBS = 1
EXPECTED_TOTAL = 40
CATEGORIES = {"hook": 15, "component": 15, "react19": 10}
PROSE = ("I would build this with a small custom hook and an effect that "
         "cleans up after itself, then wire it into the component. Let me "
         "know if you want me to write the code.")


def check(ok: bool, name: str, detail: str = "") -> bool:
    _results.append((bool(ok), name, detail))
    return bool(ok)


def all_tasks() -> list[dict]:
    return grade_react.load_task_dirs() if DIRS else grade_react.load_tasks()


def tasks() -> list[dict]:
    return [t for t in all_tasks() if not ONLY or t["id"].startswith(ONLY)]


def fenced(code: str, tag: str = "tsx") -> str:
    return (f"Here is the implementation.\n\n```{tag}\n{code.rstrip()}\n```\n\n"
            "It follows the requested signature exactly.")


def read(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def tdir(t: dict) -> str:
    return os.path.dirname(grade._p(t["reference"]))


# ---------------------------------------------------------------------------

def test_toolchain_is_alive():
    why = grade_react.ready()
    check(why is None, "react sandbox holds every pinned package", why or "")
    for pkg, ver in grade_react.PINS.items():
        got = grade._read_json(os.path.join(grade_react.NODE_MODULES, pkg,
                                            "package.json")).get("version")
        check(got == ver, f"{pkg} is {ver}", str(got))
    env_pkg = grade._read_json(os.path.join(grade_react.ENV_DIR, "package.json"))
    check(env_pkg.get("dependencies") == grade_react.PINS,
          "env/react/package.json pins exactly grade_react.PINS")
    check(os.path.isfile(os.path.join(grade_react.ENV_DIR, "package-lock.json")),
          "env/react/package-lock.json is tracked")
    r = grade.run([grade.NODE, "-e", "console.log(require('react').version,"
                   "require('react-dom').version)"], grade_react.SANDBOX, 30)
    check(re.fullmatch(r"19\.2\.\d+ 19\.2\.\d+", r["out"].strip()) is not None,
          "react and react-dom load at runtime and are 19.2.x",
          r["out"] + r["err"])
    print(f"    runtime react/react-dom: {r['out'].strip()}")
    r = grade.run([grade.NODE, grade_react.TSC, "--version"], grade_react.SANDBOX, 60)
    check("7.0.2" in r["out"], "tsc answers and is 7.0.2", r["out"] + r["err"])
    check(grade.extract("```tsx\nexport const A = () => <div/>;\n```", "react",
                        ["A"])["ok"], "a ```tsx block is react code")
    two = ("```tsx\nexport function A() { return <p>1</p> }\n```\n"
           "```tsx\n<A />\n```\n"
           "```tsx\nexport function A() { return <p>2</p> }\n```\n")
    r = grade.extract(two, "react", ["A"])
    check(r["ok"] and "<p>2</p>" in r["code"] and "<A />" not in r["code"],
          "several tsx blocks: the last one declaring the entry wins (tsx parser)",
          r.get("detail", ""))


def test_task_rows_are_complete():
    ts = [t for t in all_tasks() if not ONLY or t["id"].startswith(ONLY)]
    need = ("id", "domain", "prompt", "grader", "reference", "contaminated",
            "needs_retrieval", "difficulty")
    by: dict = {}
    for t in ts:
        by[t.get("category")] = by.get(t.get("category"), 0) + 1
        miss = [k for k in need if k not in t]
        check(not miss, f"{t['id']} has every field", str(miss))
        check(t["domain"] == "react" and t["grader"]["kind"] == "react",
              f"{t['id']} is domain/kind react")
        check(t["contaminated"] is False, f"{t['id']} is invented here, so "
              "contaminated=false")
        check(t["needs_retrieval"] == (t.get("category") == "react19"),
              f"{t['id']} needs_retrieval only for the React 19 API tasks",
              f"category={t.get('category')} needs_retrieval={t['needs_retrieval']}")
        check(t["difficulty"] in ("easy", "medium", "hard"),
              f"{t['id']} difficulty is easy|medium|hard", t["difficulty"])
        check("```tsx" in t["prompt"], f"{t['id']} prompt asks for a ```tsx block")
        for e in t["grader"].get("entry") or []:
            check(re.search(r"`[^`]*\b" + re.escape(e) + r"\b[^`]*`", t["prompt"])
                  is not None, f"{t['id']} prompt names export {e} in code")
        check(os.path.isfile(grade._p(t["reference"])), f"{t['id']} reference exists")
        check(os.path.isfile(grade._p(t["grader"]["test"])), f"{t['id']} test exists")
        d = tdir(t)
        alts = glob.glob(os.path.join(d, "alt_*.tsx"))
        wt = glob.glob(os.path.join(d, "wrong_test*.tsx"))
        check(len(alts) >= 1, f"{t['id']} has >=1 alternative correct answer", str(len(alts)))
        check(len(wt) >= 2, f"{t['id']} has >=2 compiling-but-wrong answers", str(len(wt)))
    ids = [t["id"] for t in ts]
    check(len(ids) == len(set(ids)), "task ids are unique")
    if not ONLY:
        check(len(ts) == EXPECTED_TOTAL, f"{EXPECTED_TOTAL} react tasks", str(len(ts)))
        check(by == CATEGORIES, f"category mix is {CATEGORIES}", str(by))
    print(f"    tasks per category: {by}  total {len(ts)}")


def _grade_file(t: dict, path: str, what: str, tag: str = "tsx") -> dict:
    r = grade.grade(t, fenced(read(path), tag))
    TIMES.append({"id": t["id"], "what": what, "seconds": r.get("seconds", 0.0),
                  "timings": r.get("timings"), "passed": r["passed"],
                  "stage": r["stage"]})
    return r


def _jobs_for(t: dict) -> list[tuple[str, str, str | None]]:
    """(label, path, expected) where expected None = must pass."""
    d = tdir(t)
    out = [("reference", grade._p(t["reference"]), None)]
    out += [(os.path.basename(a), a, None)
            for a in sorted(glob.glob(os.path.join(d, "alt_*.tsx")))]
    for w in sorted(glob.glob(os.path.join(d, "wrong_*.tsx"))):
        m = re.match(r"wrong_(compile|test)(?:_\w+)?\.tsx$", os.path.basename(w))
        out.append((os.path.basename(w), w, m.group(1) if m else "?"))
    return out


def _run_task(t: dict) -> list[tuple[bool, str, str]]:
    res = []
    for label, path, want in _jobs_for(t):
        if want == "?":
            res.append((False, f"{t['id']}/{label} names its expected stage", ""))
            continue
        r = _grade_file(t, path, label)
        if want is None:
            res.append((r["passed"], f"{t['id']}/{label} passes ({r['seconds']}s)",
                        f"stage={r['stage']} {r['detail'][:1500]}"))
        else:
            res.append((not r["passed"] and r["stage"] == want and not r["error"],
                        f"{t['id']}/{label} fails at {want} ({r['seconds']}s)",
                        f"passed={r['passed']} stage={r['stage']} {r['detail'][:800]}"))
    r = grade.grade(t, PROSE)
    res.append((not r["passed"] and r["stage"] == "extract",
                f"{t['id']} prose-only reply fails at extract", r["stage"]))
    r = grade.grade(t, fenced(read(grade._p(t["reference"])), "python"))
    res.append((not r["passed"] and r["stage"] == "extract",
                f"{t['id']} reference in a ```python fence fails at extract",
                r["stage"]))
    return res


def test_every_answer_grades_as_labelled():
    ts = tasks()
    if JOBS > 1:
        with ThreadPoolExecutor(JOBS) as ex:
            outs = list(ex.map(_run_task, ts))
    else:
        outs = [_run_task(t) for t in ts]
    for o in outs:
        for ok, name, detail in o:
            check(ok, name, detail)
    refs = [x for x in TIMES if x["what"] == "reference"]
    npass = sum(1 for x in refs if x["passed"])
    check(npass == len(refs), f"{npass}/{len(refs)} references pass (must be 100%)")


def report_timings() -> None:
    if not TIMES:
        return
    print(f"\n{'=' * 70}\n  grading time per task (seconds; one grade = tsc + vitest)")
    ids = sorted({x["id"] for x in TIMES})
    for i in ids:
        xs = [x for x in TIMES if x["id"] == i]
        ref = next((x["seconds"] for x in xs if x["what"] == "reference"), 0.0)
        secs = [x["seconds"] for x in xs]
        print(f"  {i:9s} reference {ref:5.2f}   n={len(xs)}  "
              f"median {statistics.median(secs):5.2f}  max {max(secs):5.2f}")
    full = [x["seconds"] for x in TIMES if (x["timings"] or {}).get("vitest")]
    tsc_only = [x["seconds"] for x in TIMES if not (x["timings"] or {}).get("vitest")]
    if full:
        print(f"  tsc+vitest grades n={len(full)}  median {statistics.median(full):.2f}"
              f"  mean {statistics.mean(full):.2f}  max {max(full):.2f}")
    if tsc_only:
        print(f"  compile-stage grades n={len(tsc_only)}  median "
              f"{statistics.median(tsc_only):.2f}")
    print(f"  total grading time: {sum(x['seconds'] for x in TIMES):.1f}s over "
          f"{len(TIMES)} calls (jobs={JOBS})")
    path = os.path.join(grade.WORK, "grade_react_timings.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for x in TIMES:
            f.write(json.dumps(x) + "\n")
    print(f"  per-call timings: {path}")


def main() -> int:
    global ONLY, DIRS, JOBS
    DIRS = "--dirs" in sys.argv
    if "--only" in sys.argv:
        ONLY = sys.argv[sys.argv.index("--only") + 1]
    if "--jobs" in sys.argv:
        JOBS = int(sys.argv[sys.argv.index("--jobs") + 1])
    t0 = time.perf_counter()
    for fn in (test_toolchain_is_alive,
               test_task_rows_are_complete,
               test_every_answer_grades_as_labelled):
        print(f"\n--- {fn.__name__} ---", flush=True)
        n0 = len(_results)
        try:
            fn()
        except Exception:                                        # noqa: BLE001
            check(False, f"{fn.__name__} itself raised",
                  traceback.format_exc().strip().split("\n")[-1])
        for ok, name, detail in _results[n0:]:
            print(("  pass  " if ok else "  FAIL  ") + name
                  + (f"   <- {detail}" if not ok and detail else ""), flush=True)
    report_timings()
    passed = sum(1 for ok, _, _ in _results if ok)
    total = len(_results)
    print(f"\n{'=' * 70}\n  {passed}/{total} checks passed "
          f"in {time.perf_counter() - t0:.0f}s")
    if passed < total:
        print("  A failing check here is a grader or task bug, not a model result.")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
