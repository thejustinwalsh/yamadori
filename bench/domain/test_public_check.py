#!/usr/bin/env python
"""Proof that grade.public_check -- the `check_solution` tool of the self-check
arms (run.py S0/S5/S6) -- compiles the answer and reveals nothing hidden.

    python bench/domain/test_public_check.py
    python bench/domain/test_public_check.py --only rs     # task id prefix

Over every task of every suite (core tasks.jsonl, tasks_react.jsonl,
tasks_type_challenges.jsonl):

  (a) the reference                       -> ok
  (b) every wrong_compile*                -> NOT ok, stage compile
  (c) every wrong_test*  (wrong only by   -> ok. This is the proof that no
      the hidden tests)                      hidden test reaches the check: an
                                             answer only a hidden test rejects
                                             must look clean here.
  (d) tc: the upstream template (a        -> ok (it compiles; only the hidden
      placeholder, usually `= any`)          test cases reject it)

None may come back as a checker error. Also: every error line is free of the
hidden test file's text; the public sandbox holds no test file after a check;
a reply with the code in a fence is extracted as grade() would; prose is not
ok; a tc answer missing a template name is not ok.
"""
from __future__ import annotations

import glob
import os
import re
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import grade  # noqa: E402

_results: list[tuple[bool, str, str]] = []
ONLY: str | None = None


def check(ok: bool, name: str, detail: str = "") -> bool:
    _results.append((bool(ok), name, detail))
    return bool(ok)


def read(p: str) -> str:
    with open(p, encoding="utf-8") as f:
        return f.read()


def suites() -> list[dict]:
    ts = grade.load_tasks()
    for p in (os.path.join(HERE, "tasks_react.jsonl"), grade.TC_JSONL):
        if os.path.isfile(p):
            ts += grade.load_tasks(p)
    return [t for t in ts if not ONLY or t["id"].startswith(ONLY)]


def hidden_lines(task: dict) -> list[str]:
    """Distinctive lines of every file the grader holds back."""
    g = task["grader"]
    out = []
    for k in ("test", "host_test", "wasm_test"):
        if g.get(k) and os.path.isfile(grade._p(g[k])):
            for ln in read(grade._p(g[k])).splitlines():
                s = ln.strip()
                if len(s) >= 24 and not s.startswith(("//", "import ", "use ", "*")):
                    out.append(s)
    for r in g.get("require") or []:
        out.append(f"missing the r185 API: {r['name']}")
    return out


def leaks(task: dict, res: dict) -> list[str]:
    text = "\n".join(res.get("errors") or [])
    return [h for h in hidden_lines(task) if h in text]


def test_every_task() -> None:
    n = {"ref": 0, "wc": 0, "wt": 0, "tpl": 0}
    bad = {"ref": [], "wc": [], "wt": [], "tpl": [], "err": [], "leak": []}
    for t in suites():
        d = os.path.dirname(grade._p(t["reference"]))
        cases = [("ref", t["reference"])]
        cases += [("wc", p) for p in sorted(glob.glob(os.path.join(d, "wrong_compile*")))]
        cases += [("wt", p) for p in sorted(glob.glob(os.path.join(d, "wrong_test*")))]
        if t["grader"].get("kind") == "tc_tests" and t["grader"].get("template"):
            cases.append(("tpl", t["grader"]["template"]))
        for kind, path in cases:
            r = grade.public_check(t, read(grade._p(path)))
            n[kind] += 1
            name = f"{t['id']}/{os.path.basename(path)}"
            if r["checker_error"]:
                bad["err"].append(f"{name}: {r['errors'][:2]}")
                continue
            want_ok = kind != "wc"
            if r["ok"] != want_ok or (kind == "wc" and r["stage"] != "compile"):
                bad[kind].append(f"{name}: ok={r['ok']} stage={r['stage']} "
                                 f"{(r['errors'] or [''])[0][:160]}")
            lk = leaks(t, r)
            if lk:
                bad["leak"].append(f"{name}: {lk[0][:80]}")
    check(n["ref"] > 0 and not bad["ref"],
          f"every reference passes public_check ({n['ref']} tasks)",
          "; ".join(bad["ref"][:6]))
    check(n["wc"] > 0 and not bad["wc"],
          f"every wrong_compile answer fails it at compile ({n['wc']} answers)",
          "; ".join(bad["wc"][:6]))
    check(n["wt"] > 0 and not bad["wt"],
          f"every wrong_test answer PASSES it: nothing hidden reaches the check "
          f"({n['wt']} answers)", "; ".join(bad["wt"][:6]))
    if n["tpl"]:
        check(not bad["tpl"],
              f"every type-challenges template (a placeholder) passes it "
              f"({n['tpl']} templates)", "; ".join(bad["tpl"][:6]))
    check(not bad["err"], "no checker error on any answer",
          "; ".join(bad["err"][:6]))
    check(not bad["leak"], "no error line quotes a hidden test file",
          "; ".join(bad["leak"][:6]))


def _task(tid: str) -> dict | None:
    return next((t for t in suites() if t["id"] == tid), None)


def test_sandbox_holds_no_hidden_file() -> None:
    for tid in ("ts01", "tg01", "react-01", "tc-00002-return-type", "rs01", "rs02"):
        t = _task(tid)
        if t is None:
            continue
        r = grade.public_check(t, read(grade._p(t["reference"])))
        kind = t["grader"]["kind"]
        if kind == "rust":
            box = os.path.join(grade.RUST_DIR, "t_" + grade._safe(tid), "src")
        elif kind == "react":
            import grade_react
            box = os.path.join(grade_react.SANDBOX, "public", grade._safe(tid))
        else:
            box = os.path.join(grade.NODE_DIR, "public", grade._safe(tid))
        files = sorted(os.listdir(box)) if os.path.isdir(box) else []
        hidden = [f for f in files if re.search(r"test|grader", f, re.I)]
        check(r["ok"] and files and not hidden,
              f"{tid}: the public sandbox holds no test or grader file",
              f"ok={r['ok']} files={files}")


def test_shapes() -> None:
    t = _task("ts01")
    if t:
        ref = read(grade._p(t["reference"]))
        r = grade.public_check(t, f"Here it is.\n\n```ts\n{ref}\n```\n")
        check(r["ok"], "a fenced reply is extracted and checked like grade() does")
        r = grade.public_check(t, "```python\nprint(1)\n```")
        check(not r["ok"] and r["stage"] == "extract",
              "code in a wrong-language fence: stage extract", str(r["errors"]))
        r = grade.public_check(t, "")
        check(not r["ok"] and r["stage"] == "extract", "empty code: stage extract")
        r = grade.public_check(t, ref + "\nconst broken: number = 'x';\n")
        check(not r["ok"] and r["n_errors"] >= 1
              and all("solution.ts" in e or e.startswith(" ") for e in r["errors"]),
              "a type error is reported against solution.ts only",
              str(r["errors"][:3]))
        check(not any(grade.WORK in e or grade.WORK.replace("\\", "/") in e
                      for e in r["errors"]),
              "no sandbox path in the errors", str(r["errors"][:2]))
    t = _task("tc-00002-return-type")
    if t:
        r = grade.public_check(t, "type Unrelated = 1")
        check(not r["ok"] and any("MyReturnType" in e for e in r["errors"]),
              "tc: a template name the answer does not declare is reported",
              str(r["errors"]))
    r = grade.public_check({"id": "x", "grader": {"kind": "nope"}}, "code")
    check(r["checker_error"] and not r["ok"],
          "an unknown grader kind is a checker error, not a code failure")
    big = "\n".join(f"let v{i}: number = 'x';" for i in range(200))
    if t:
        r = grade.public_check(_task("ts01"), big)
        check(r["n_errors"] >= 200 and len(r["errors"]) <= grade.PUBLIC_MAX_LINES
              and r["truncated"],
              f"errors trimmed to the first {grade.PUBLIC_MAX_LINES} lines",
              f"n_errors={r['n_errors']} shown={len(r['errors'])}")


def main() -> int:
    global ONLY
    a = sys.argv[1:]
    if "--only" in a:
        ONLY = a[a.index("--only") + 1]
    t0 = time.perf_counter()
    for fn in (test_every_task, test_sandbox_holds_no_hidden_file, test_shapes):
        print(f"\n--- {fn.__name__} ---", flush=True)
        n0 = len(_results)
        try:
            fn()
        except Exception:                                        # noqa: BLE001
            check(False, f"{fn.__name__} itself raised",
                  traceback.format_exc().strip().split("\n")[-1])
        for ok, nm, detail in _results[n0:]:
            print(("  pass  " if ok else "  FAIL  ") + nm
                  + (f"\n        <- {detail}" if not ok and detail else ""))
    passed = sum(1 for ok, _, _ in _results if ok)
    print(f"\n{passed}/{len(_results)} checks passed "
          f"in {time.perf_counter() - t0:.1f}s")
    return 0 if passed == len(_results) else 1


if __name__ == "__main__":
    sys.exit(main())
