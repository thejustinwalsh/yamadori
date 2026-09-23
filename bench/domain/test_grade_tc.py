#!/usr/bin/env python
"""Proof that the type-challenges grader (kind `tc_tests`) is sound.

Sibling of test_grade.py, for tasks_type_challenges.jsonl. Upstream ships no
solutions, so every included challenge carries a reference.ts written for
this repo, and each one is run through `grade.grade` as a model reply:

  (a) reference.ts in a ```ts fence              -> must PASS
  (b) reference.ts written as a module (a
      trailing `export {}`), in a ```ts fence    -> must PASS (the module path)
  (c) upstream template.ts -- the placeholder,
      usually `= any`                            -> must FAIL at compile|test
  (d) an empty implementation (a fence holding
      only a comment)                            -> must FAIL at compile|test
  (e) prose with no code block                   -> must FAIL at extract
  (f) reference.ts in a ```python fence          -> must FAIL at extract

None may come back as stage `error`. 100% of included references must pass.
A challenge whose reference cannot be made to pass is removed from the set
and listed in tasks_type_challenges/excluded.json with the reason -- never
shipped ungradable.

It also proves the ground is alive: tsc answers and is the pinned 7.0.2, the
vendored @type-challenges/utils is byte-identical to upstream (when the
checkout is present), and `Equal` can fail (a check that cannot fail is not a
check).

    python bench/domain/test_grade_tc.py                 # everything
    python bench/domain/test_grade_tc.py --only tc-0000  # id prefix
    python bench/domain/test_grade_tc.py --refs          # (a) only, fast
    python bench/domain/test_grade_tc.py --dirs          # rows from the task
                                                         # dirs, not the jsonl
    python bench/domain/test_grade_tc.py --dirs --quick --only ID --verbose
                                                         # authoring one task
    python bench/domain/test_grade_tc.py --jobs 4        # parallel; timings
                                                         # are then contended
"""
from __future__ import annotations

import filecmp
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

_results: list[tuple[bool, str, str]] = []
TIMES: list[dict] = []
ONLY: str | None = None
REFS_ONLY = False
JOBS = 1
PROSE = ("I would approach this by first working out the recursive structure "
         "of the type and then writing a conditional type. Let me know if you "
         "want me to write the code.")
EMPTY = "Here is my answer.\n\n```ts\n// TODO: implement\n```\n"


def check(ok: bool, name: str, detail: str = "") -> bool:
    _results.append((bool(ok), name, detail))
    return bool(ok)


def read(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def fenced(code: str, tag: str = "ts") -> str:
    return (f"Here is the implementation.\n\n```{tag}\n{code.rstrip()}\n```\n\n"
            "It follows the requested signature exactly.")


DIRS = False


def tasks() -> list[dict]:
    if DIRS:   # authoring: rows straight from the task dirs, jsonl untouched
        import build_type_challenges
        ts = build_type_challenges.collect(write_files=False)["rows"]
    else:
        ts = grade.load_tasks(grade.TC_JSONL)
    return [t for t in ts if not ONLY or t["id"].startswith(ONLY)]


# ---------------------------------------------------------------------------

def test_ground_is_alive():
    check(grade.node_ready() is None, "node sandbox has the pinned packages",
          grade.node_ready() or "")
    r = grade.run([grade.NODE, grade.TSC, "--version"], grade.NODE_DIR, 60)
    check("7.0.2" in r["out"], "tsc answers and is 7.0.2", r["out"] + r["err"])
    u = os.path.join(grade.TC_UTILS, "index.d.ts")
    check(os.path.isfile(u), "vendored @type-challenges/utils exists", u)
    up = os.path.join(grade.WORK, "tc-src", "utils", "index.d.ts")
    if os.path.isfile(up):
        check(filecmp.cmp(u, up, shallow=False),
              "vendored utils are byte-identical to the upstream checkout")
    lic = os.path.join(grade.TC_DIR, "LICENSE.type-challenges")
    check(os.path.isfile(lic) and "MIT License" in read(lic),
          "the upstream MIT licence is carried alongside the tasks")
    check(os.path.isfile(os.path.join(grade.TC_DIR, "NOTICE")), "NOTICE exists")
    # Equal must be able to fail, and a module answer must be importable.
    t = {"id": "tc-selftest", "grader": {
        "kind": "tc_tests", "entry": ["Id"],
        "test": os.path.join(grade.WORK, "tc_selftest", "test-cases.ts")}}
    os.makedirs(os.path.dirname(t["grader"]["test"]), exist_ok=True)
    body = ("import type { Equal, Expect } from '@type-challenges/utils'\n"
            "type cases = [Expect<Equal<Id<1>, 1>>]\n")
    if not os.path.isfile(t["grader"]["test"]) or read(t["grader"]["test"]) != body:
        with open(t["grader"]["test"], "w", encoding="utf-8", newline="\n") as f:
            f.write(body)
    r = grade.grade(t, fenced("type Id<T> = T"))
    check(r["passed"], "self-test: a correct answer passes", r["detail"])
    r = grade.grade(t, fenced("export type Id<T> = T"))
    check(r["passed"], "self-test: the same answer as an exported module "
          "passes", r["detail"])
    r = grade.grade(t, fenced("import type { Equal } from '@type-challenges/utils'"
                              "\ntype Id<T> = Equal<T, T> extends true ? T : never"))
    check(r["passed"], "self-test: an answer that imports the utils passes",
          r["detail"])
    r = grade.grade(t, fenced("type Id<T> = 2"))
    check(not r["passed"] and r["stage"] == "test",
          "self-test: a wrong answer fails at test (Equal can fail)",
          f"{r['stage']} {r['detail']}")
    r = grade.grade(t, fenced("type Id<T> = T;;; let x: number = 'a'"))
    check(not r["passed"] and r["stage"] == "compile",
          "self-test: an answer with its own type error fails at compile",
          f"{r['stage']} {r['detail']}")
    r = grade.grade(t, fenced("type Id<T> = any"))
    check(not r["passed"], "self-test: `any` is not Equal to 1", r["stage"])


def _suppressions(code: str) -> list[str]:
    """@ts-ignore / @ts-expect-error / @ts-nocheck comments, found by the
    parser (a directive-looking string literal is not one)."""
    out, stack = [], [grade.parse(code, "typescript")]
    while stack:
        n = stack.pop()
        if n.type == "comment" and re.search(
                r"@ts-(ignore|expect-error|nocheck)", grade._txt(n)):
            out.append(f"line {n.start_point[0] + 1}: {grade._txt(n)[:60]}")
        stack.extend(n.children)
    return out


def test_rows_are_complete():
    ts = tasks()
    need = ("id", "domain", "prompt", "grader", "reference", "contaminated",
            "needs_retrieval", "difficulty")
    by = {}
    for t in ts:
        by[t["difficulty"]] = by.get(t["difficulty"], 0) + 1
        miss = [k for k in need if k not in t]
        check(not miss, f"{t['id']} has every field", str(miss))
        check(re.fullmatch(r"tc-\d{5}-[a-z0-9-]+", t["id"]) is not None,
              f"{t['id']} keeps the upstream number and slug")
        check(t["contaminated"] is True and t["domain"] == "typescript_tc"
              and t["grader"]["kind"] == "tc_tests",
              f"{t['id']} is contaminated typescript_tc / tc_tests")
        for k in ("test", "template"):
            check(os.path.isfile(grade._p(t["grader"][k])),
                  f"{t['id']} {k} file exists")
        check(os.path.isfile(grade._p(t["reference"])), f"{t['id']} reference exists")
        check(bool(t["grader"]["entry"]), f"{t['id']} has entry names")
        ref = read(grade._p(t["reference"]))
        nm = grade.tc_names(ref)
        check(not nm["module"], f"{t['id']} reference is a global script, as "
              "the upstream playground expects")
        miss = sorted(set(t["grader"]["entry"]) - nm["declared"])
        check(not miss, f"{t['id']} reference declares every entry name", str(miss))
        sup = _suppressions(ref)
        check(not sup, f"{t['id']} reference suppresses no compiler error",
              "; ".join(sup))
        p = t["prompt"]
        tpl = read(grade._p(t["grader"]["template"])).rstrip()
        check("```ts" in p and tpl in p, f"{t['id']} prompt carries the template "
              "and asks for a ```ts block")
        check("tsch.js.org" not in p and "/solutions" not in p
              and "Share your Solutions" not in p,
              f"{t['id']} prompt carries no link to solutions")
    ids = [t["id"] for t in ts]
    check(len(ids) == len(set(ids)), "task ids are unique")
    ex = json.loads(read(os.path.join(grade.TC_DIR, "excluded.json")))
    for k, v in ex.items():
        check(isinstance(v, str) and len(v) > 20, f"exclusion {k} states a reason")
        check(k not in ids, f"excluded {k} is not shipped")
    up = os.path.join(grade.WORK, "tc-src", "questions")
    if os.path.isdir(up) and not ONLY:
        n_up = len([d for d in os.listdir(up) if re.match(r"\d{5}-", d)])
        check(len(ids) + len(ex) == n_up, "every upstream challenge is either "
              f"included or excluded with a reason ({len(ids)} + {len(ex)} "
              f"of {n_up})")
    print(f"    included per difficulty: {by}  total {len(ts)}; excluded {len(ex)}")


def _time(t: dict, what: str, r: dict) -> None:
    TIMES.append({"id": t["id"], "difficulty": t["difficulty"], "what": what,
                  "seconds": r.get("seconds", 0.0), "timings": r.get("timings")})


def _prove(t: dict) -> list[tuple[bool, str, str]]:
    out = []
    ref = read(grade._p(t["reference"]))
    r = grade.grade(t, fenced(ref))
    _time(t, "reference", r)
    out.append((r["passed"], f"{t['id']} reference passes ({r['seconds']}s)",
                f"stage={r['stage']} {r['detail'][:900]}"))
    if REFS_ONLY:
        return out
    r = grade.grade(t, fenced(ref.rstrip() + "\n\nexport {}\n"))
    _time(t, "reference_module", r)
    out.append((r["passed"], f"{t['id']} reference as a module passes",
                f"stage={r['stage']} {r['detail'][:600]}"))
    r = grade.grade(t, fenced(read(grade._p(t["grader"]["template"]))))
    _time(t, "template", r)
    out.append((not r["passed"] and r["stage"] in ("compile", "test")
                and not r["error"], f"{t['id']} upstream placeholder template "
                f"fails at {r['stage']}", f"passed={r['passed']} "
                f"stage={r['stage']} {r['detail'][:300]}"))
    r = grade.grade(t, EMPTY)
    _time(t, "empty", r)
    out.append((not r["passed"] and r["stage"] in ("compile", "test")
                and not r["error"], f"{t['id']} empty implementation fails at "
                f"{r['stage']}", f"passed={r['passed']} stage={r['stage']}"))
    r = grade.grade(t, PROSE)
    _time(t, "prose", r)
    out.append((not r["passed"] and r["stage"] == "extract",
                f"{t['id']} prose-only reply fails at extract", r["stage"]))
    r = grade.grade(t, fenced(ref, "python"))
    _time(t, "wrong_fence", r)
    out.append((not r["passed"] and r["stage"] == "extract",
                f"{t['id']} reference in a ```python fence fails at extract",
                r["stage"]))
    return out


def test_every_challenge_is_provably_gradable():
    ts = tasks()
    if JOBS > 1:
        with ThreadPoolExecutor(JOBS) as ex:
            for res in ex.map(_prove, ts):
                _results.extend(res)
    else:
        for t in ts:
            _results.extend(_prove(t))


def report_timings() -> None:
    if not TIMES:
        return
    print(f"\n{'=' * 70}\n  grading time per call (seconds, jobs={JOBS})")
    for what in ("reference", "reference_module", "template", "empty",
                 "prose", "wrong_fence"):
        xs = [x["seconds"] for x in TIMES if x["what"] == what]
        if xs:
            print(f"  {what:16s} n={len(xs):3d}  median {statistics.median(xs):6.2f}"
                  f"  mean {statistics.mean(xs):6.2f}  max {max(xs):6.2f}")
    for d in ("warm", "easy", "medium", "hard", "extreme"):
        xs = [x["seconds"] for x in TIMES if x["difficulty"] == d
              and x["what"] == "reference"]
        if xs:
            print(f"  reference/{d:8s} n={len(xs):3d}  median "
                  f"{statistics.median(xs):6.2f}  max {max(xs):6.2f}")
    worst = sorted(TIMES, key=lambda x: -x["seconds"])[:5]
    print("  slowest:", ", ".join(f"{x['id']}/{x['what']} {x['seconds']}s"
                                  for x in worst))
    print(f"  total grading time: {sum(x['seconds'] for x in TIMES):.1f}s "
          f"over {len(TIMES)} calls")
    if "--quick" in sys.argv:
        return
    path = os.path.join(grade.WORK, "grade_timings_tc.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for x in TIMES:
            f.write(json.dumps(x) + "\n")
    print(f"  per-call timings: {path}")


def main() -> int:
    global ONLY, REFS_ONLY, JOBS, DIRS
    REFS_ONLY = "--refs" in sys.argv
    DIRS = "--dirs" in sys.argv
    if "--only" in sys.argv:
        ONLY = sys.argv[sys.argv.index("--only") + 1]
    if "--jobs" in sys.argv:
        JOBS = int(sys.argv[sys.argv.index("--jobs") + 1])
    t0 = time.perf_counter()
    fns = [test_ground_is_alive, test_rows_are_complete,
           test_every_challenge_is_provably_gradable]
    if "--quick" in sys.argv:     # authoring one challenge: proof only
        fns = fns[-1:]
    for fn in fns:
        print(f"\n--- {fn.__name__} ---", flush=True)
        n0 = len(_results)
        try:
            fn()
        except Exception:                                        # noqa: BLE001
            check(False, f"{fn.__name__} itself raised",
                  traceback.format_exc().strip().split("\n")[-1])
        for ok, name, detail in _results[n0:]:
            if not ok or "--verbose" in sys.argv:
                print(("  pass  " if ok else "  FAIL  ") + name
                      + (f"   <- {detail}" if not ok and detail else ""),
                      flush=True)
        print(f"  {sum(1 for r in _results[n0:] if r[0])}/"
              f"{len(_results) - n0} passed", flush=True)
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
