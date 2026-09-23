#!/usr/bin/env python
"""Proof that the domain graders are sound, before any model is scored by them.

A grader bug scored as a model failure is exactly the error this repo keeps
making (PROTOCOL rules 3 and 14). So every task's grader is run here on:

  (a) its reference solution               -> must PASS
  (b) every deliberately wrong answer      -> must FAIL, at the stage its file
      in the task directory, named           name says (wrong_compile.* fails
      wrong_<stage>[_n].<ext>                at `compile`, wrong_test.* at `test`)
  (c) every alternative correct answer,
      alt_*.<ext>                          -> must PASS (no false negatives)
  (d) a reply with no code block           -> must fail at `extract`
  (e) the reference in a wrong-language
      fence                                -> must fail at `extract`

None of those may come back as stage `error` -- that would be the grader, not
the answer. 100% of references must pass; one reference failing fails the run.

It also proves the ground under the graders is alive (rule 1): the pinned
toolchains answer, typegpu in node_modules is byte-identical to the indexed
typegpu@0.12.5 source, the three@0.185.1 export lists are non-trivial and
agree with the package index database, and every name a TSL task forbids is
either absent from r185 or carries a deprecation marker in its source.

    python bench/domain/test_grade.py              # everything
    python bench/domain/test_grade.py --only rs    # task ids with this prefix
    python bench/domain/test_grade.py --dirs       # read tasks/*/task.json, not
                                                   # tasks.jsonl (while authoring)
"""
from __future__ import annotations

import filecmp
import glob
import json
import os
import re
import sqlite3
import statistics
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import grade  # noqa: E402

_results: list[tuple[bool, str, str]] = []
TIMES: list[dict] = []
ONLY: str | None = None

FENCE = {"ts": "ts", "rust": "rust", "tsl": "js"}
WRONG_FENCE = {"ts": "python", "rust": "ts", "tsl": "rust"}
PROSE = ("I would approach this by first reading the documentation for the "
         "relevant API and then writing a function with the requested "
         "signature. Let me know if you want me to write the code.")


def check(ok: bool, name: str, detail: str = "") -> bool:
    _results.append((bool(ok), name, detail))
    return bool(ok)


DIRS = False


def all_tasks() -> list[dict]:
    return grade.load_task_dirs() if DIRS else grade.load_tasks()


def tasks() -> list[dict]:
    ts = all_tasks()
    return [t for t in ts if not ONLY or t["id"].startswith(ONLY)]


def fenced(code: str, tag: str) -> str:
    return (f"Here is the implementation.\n\n```{tag}\n{code.rstrip()}\n```\n\n"
            "It follows the requested signature exactly.")


def read(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------------------

def test_extraction_rules():
    ex = grade.extract
    check(not ex(PROSE, "ts", ["f"])["ok"], "prose with no fence is not code")
    check(not ex("```python\nprint(1)\n```", "ts", [])["ok"],
          "a block in another language is not accepted")
    r = ex("```\nexport const f = 1;\n```", "ts", ["f"])
    check(r["ok"] and "export const f" in r["code"], "an untagged block is accepted")
    r = ex("```typescript\nexport const f = 1;\n```", "ts", ["f"])
    check(r["ok"], "```typescript is a ts block")
    r = ex("~~~rs\npub fn f() {}\n~~~", "rust", ["f"])
    check(r["ok"] and "pub fn f" in r["code"], "~~~ fences and the rs alias")
    two = ("First try:\n```ts\nexport function f() { return 1 }\n```\n"
           "Usage:\n```ts\nconsole.log(f())\n```\n"
           "Fixed:\n```ts\nexport function f() { return 2 }\n```\n")
    r = ex(two, "ts", ["f"])
    check(r["ok"] and "return 2" in r["code"] and "return 1" not in r["code"]
          and "console.log" not in r["code"],
          "several blocks: the last one declaring every entry wins", r.get("detail", ""))
    split = ("```rust\n#[repr(C)] pub struct S { pub a: u8 }\n```\n"
             "```rust\n#[no_mangle] pub extern \"C\" fn g() -> usize { 1 }\n```\n")
    r = ex(split, "rust", ["S", "g"])
    check(r["ok"] and "struct S" in r["code"] and "fn g" in r["code"],
          "entries split across blocks: all blocks are concatenated", r.get("detail", ""))
    r = ex("```ts\nexport const f = 1;\n", "ts", ["f"])
    check(r["ok"] and r.get("truncated"), "an unterminated fence is extracted and flagged")
    r = ex("````md\n```ts\nx\n```\n````", "ts", [])
    check(not r["ok"], "a ts fence nested inside a 4-backtick md block is not code")
    check(grade.declared_names("export type A = 1; interface B {}; "
                               "export const c = 1, d = 2; function e() {}",
                               "ts") == {"A", "B", "c", "d", "e"},
          "ts top-level declarations are read by the parser")
    check(grade.declared_names("#[no_mangle] pub extern \"C\" fn f() {}\n"
                               "pub struct S; extern \"C\" { fn g(); }", "rust")
          == {"f", "S", "g"}, "rust top-level declarations are read by the parser")
    r = grade.grade({"id": "x", "grader": {"kind": "nope"}}, "```ts\n1\n```")
    check(r["stage"] == "error" and r["error"],
          "an unknown grader kind is a grader error, not a model failure")
    lk = grade._TaskLock("__selftest__")
    with open(lk.path, "w") as f:
        f.write("999999")          # a holder that died mid-grade
    t0 = time.perf_counter()
    with lk:
        pass
    check(time.perf_counter() - t0 < 2 and not os.path.exists(lk.path),
          "a lock left by a dead process is reclaimed, not waited out")


def test_toolchains_are_alive():
    check(grade.node_ready() is None, "node sandbox has the pinned packages",
          grade.node_ready() or "")
    r = grade.run([grade.NODE, grade.TSC, "--version"], grade.NODE_DIR, 60)
    check("7.0.2" in r["out"], "tsc answers and is 7.0.2", r["out"] + r["err"])
    why = grade.rust_ready(True, True)
    check(why is None, "cargo, the wasm32 target and wasm-bindgen 0.2.128 are ready",
          why or "")
    r = grade.run([grade.RUSTC, "--version"], grade.HERE, 30)
    print(f"    rustc: {r['out'].strip()}")
    r = grade.run([grade.WASM_BINDGEN, "--version"], grade.HERE, 30)
    print(f"    wasm-bindgen: {r['out'].strip()}")
    r = grade.run([grade.NODE, "--version"], grade.HERE, 30)
    print(f"    node: {r['out'].strip()}")
    lock = read(os.path.join(grade.ENV_DIR, "Cargo.lock"))
    check('name = "wasm-bindgen"\nversion = "0.2.128"' in lock,
          "the tracked Cargo.lock pins the wasm-bindgen crate to the CLI version")


def test_typegpu_is_the_indexed_version():
    nm = os.path.join(grade.NODE_DIR, "node_modules", "typegpu")
    src = os.path.join(grade.PKG_STORE, "_src", "typegpu@0.12.5")
    check(os.path.isdir(src), "typegpu@0.12.5 source is held in the package store", src)
    if not os.path.isdir(src):
        return
    diffs, n = [], 0
    for p in glob.glob(os.path.join(src, "**", "*"), recursive=True):
        if os.path.isfile(p):
            n += 1
            q = os.path.join(nm, os.path.relpath(p, src))
            if not os.path.isfile(q) or not filecmp.cmp(p, q, shallow=False):
                diffs.append(os.path.relpath(p, src))
    check(n > 100 and not diffs,
          f"node_modules/typegpu is byte-identical to the indexed source ({n} files)",
          ", ".join(diffs[:5]))
    ver = grade._read_json(os.path.join(nm, "package.json")).get("version")
    check(ver == "0.12.5", "node_modules/typegpu/package.json says 0.12.5", str(ver))


def test_three_exports_are_alive_and_match_the_index():
    ex = grade.three_exports()
    sizes = {k: len(v) for k, v in ex.items()}
    print(f"    three@0.185.1 export counts: {sizes}")
    check(sizes["three/tsl"] > 500 and sizes["three/webgpu"] > 500
          and sizes["three"] > 300, "export lists are non-trivial", str(sizes))
    for spec, names in (("three/tsl", ("Fn", "negateOnBackSide", "normalView",
                                       "uniform", "vec3")),
                        ("three/webgpu", ("WebGPURenderer", "MeshStandardNodeMaterial")),
                        ("three", ("Mesh", "Vector3"))):
        miss = [n for n in names if n not in ex[spec]]
        check(not miss, f"{spec} exports known names", str(miss))
    check("flipOnBackSide" not in ex["three/tsl"],
          "a made-up name is not in the export list (the check can fail)")
    db = os.path.join(grade.PKG_STORE, "three@0.185.1.sqlite3")
    if not check(os.path.isfile(db), "the three@0.185.1 package index exists", db):
        return
    con = sqlite3.connect(db)
    try:
        defs = {r[0] for r in con.execute("SELECT DISTINCT name FROM defs")}
    finally:
        con.close()
    check(len(defs) > 1000, f"the package index has a live symbol table ({len(defs)} names)")
    tsl = ex["three/tsl"]
    lost = sorted(n for n in tsl if n not in defs)
    # three.tsl.js re-exports TSL.* members; a few are object members or
    # aliases the symbol table indexes under another node type.
    check(len(lost) <= 0.05 * len(tsl),
          f"three/tsl exports agree with the index's defs table "
          f"({len(tsl) - len(lost)}/{len(tsl)} present)", ", ".join(lost[:15]))


_DEPRECATION_WINDOW = 12
_src_lines: dict = {}


def _deprecated_in_source(name: str) -> str | None:
    """File:line where `name` sits within a few lines of a deprecation marker."""
    if not _src_lines:
        for p in glob.glob(os.path.join(grade.THREE_SRC, "src", "**", "*.js"),
                           recursive=True):
            _src_lines[p] = read(p).splitlines()
    pat = re.compile(r"\b" + re.escape(name) + r"\b")
    for p, lines in _src_lines.items():
        for i, line in enumerate(lines):
            if pat.search(line):
                lo, hi = max(0, i - _DEPRECATION_WINDOW), i + _DEPRECATION_WINDOW
                if any("deprecated" in l.lower() for l in lines[lo:hi]):
                    return f"{os.path.relpath(p, grade.THREE_SRC)}:{i + 1}"
    return None


def _mentioned_in_source(name: str) -> str | None:
    _deprecated_in_source("__warm__")
    pat = re.compile(r"\b" + re.escape(name) + r"\b")
    for p, lines in _src_lines.items():
        for i, line in enumerate(lines):
            if pat.search(line):
                return f"{os.path.relpath(p, grade.THREE_SRC)}:{i + 1}"
    return None


def _files_mentioning(name: str) -> list[str]:
    _deprecated_in_source("__warm__")
    pat = re.compile(r"\b" + re.escape(name) + r"\b")
    return sorted(os.path.relpath(p, grade.THREE_SRC) for p, lines
                  in _src_lines.items() if any(pat.search(l) for l in lines))


def _file_is_deprecated(rel: str) -> bool:
    """The module's leading doc comment (its class) carries @deprecated."""
    lines = _src_lines.get(os.path.join(grade.THREE_SRC, rel), [])
    return any("@deprecated" in l for l in lines[:20])


def test_tsl_ground_truth_is_version_marked():
    ex = grade.three_exports()
    for t in tasks():
        if t["grader"]["kind"] != "tsl":
            continue
        g = t["grader"]
        check(t["contaminated"] is True, f"{t['id']} is flagged contaminated")
        for req in g.get("require", []):
            if req.get("as") == "import" or req.get("from"):
                spec = req.get("from")
                pool = ex[spec] if spec in ex else set().union(*ex.values())
                check(req["name"] in pool,
                      f"{t['id']} requires {req['name']}, which r185 exports")
        for nm in g.get("forbid", []):
            exported = any(nm in v for v in ex.values())
            if exported:
                where = _deprecated_in_source(nm)
                check(where is not None, f"{t['id']} forbids {nm}: exported by r185 "
                      f"but deprecated at {where}", "no deprecation marker near it")
            else:
                # Not an export. Either a deprecated member (method/property/
                # option) kept as a warning shim, or gone from r185 entirely.
                where = _deprecated_in_source(nm)
                present = _mentioned_in_source(nm)
                homes = _files_mentioning(nm)
                if where is not None:
                    check(True, f"{t['id']} forbids {nm}: not exported; a "
                          f"deprecated member shim at {where}")
                elif homes and all(_file_is_deprecated(p) for p in homes):
                    check(True, f"{t['id']} forbids {nm}: only a member of "
                          f"deprecated module(s) {', '.join(homes)}")
                else:
                    check(present is None, f"{t['id']} forbids {nm}: absent "
                          f"from r185 source", f"still in source at {present} "
                          f"with no deprecation marker near it")


def test_task_rows_are_complete():
    ts = [t for t in all_tasks() if not ONLY or t["id"].startswith(ONLY)]
    need = ("id", "domain", "prompt", "grader", "reference", "contaminated",
            "needs_retrieval", "difficulty")
    by = {}
    for t in ts:
        by[t["domain"]] = by.get(t["domain"], 0) + 1
        miss = [k for k in need if k not in t]
        check(not miss, f"{t['id']} has every field", str(miss))
        check(os.path.isfile(grade._p(t["reference"])),
              f"{t['id']} reference exists", t["reference"])
        tdir = os.path.dirname(grade._p(t["reference"]))
        wrong = glob.glob(os.path.join(tdir, "wrong_*"))
        check(any(os.path.basename(w).startswith("wrong_test") for w in wrong),
              f"{t['id']} has a compiling-but-wrong answer (wrong_test*)")
        check(t["difficulty"] in ("easy", "medium", "hard"),
              f"{t['id']} difficulty is easy|medium|hard", t["difficulty"])
        check("```" in t["prompt"] or "code block" in t["prompt"],
              f"{t['id']} prompt asks for a code block")
        if t["domain"] == "three_tsl":
            check(t["contaminated"], f"{t['id']} (three.js) is contaminated")
        g = t["grader"]
        for k in ("test", "host_test", "wasm_test"):
            if k in g:
                check(os.path.isfile(grade._p(g[k])), f"{t['id']} grader file {k} exists")
    ids = [t["id"] for t in ts]
    check(len(ids) == len(set(ids)), "task ids are unique")
    print(f"    tasks per domain: {by}  total {len(ts)}")


def _time(t: dict, what: str, r: dict) -> None:
    TIMES.append({"id": t["id"], "kind": t["grader"]["kind"], "what": what,
                  "seconds": r.get("seconds", 0.0), "timings": r.get("timings")})


def test_every_reference_passes():
    for t in tasks():
        kind = t["grader"]["kind"]
        r = grade.grade(t, fenced(read(grade._p(t["reference"])), FENCE[kind]))
        _time(t, "reference", r)
        check(r["passed"], f"{t['id']} reference passes ({r['seconds']}s)",
              f"stage={r['stage']} {r['detail'][:1200]}")


def test_every_alternative_correct_answer_passes():
    """alt_*.<ext>: a correct answer written differently from the reference.

    The other half of soundness. A grader that only accepts the reference's
    exact shape scores a correct model as wrong, which is the same bug as
    scoring its own crash as a model failure.
    """
    n = 0
    for t in tasks():
        kind = t["grader"]["kind"]
        tdir = os.path.dirname(grade._p(t["reference"]))
        for a in sorted(glob.glob(os.path.join(tdir, "alt_*"))):
            n += 1
            r = grade.grade(t, fenced(read(a), FENCE[kind]))
            _time(t, "reference", r)
            check(r["passed"], f"{t['id']}/{os.path.basename(a)} passes "
                  f"({r['seconds']}s)", f"stage={r['stage']} {r['detail'][:1200]}")
    print(f"    {n} alternative correct answers")


def test_every_wrong_answer_fails_at_its_stage():
    for t in tasks():
        kind = t["grader"]["kind"]
        tdir = os.path.dirname(grade._p(t["reference"]))
        for w in sorted(glob.glob(os.path.join(tdir, "wrong_*"))):
            base = os.path.basename(w)
            m = re.match(r"wrong_(compile|test)(?:_\w+)?\.\w+$", base)
            if not check(m is not None, f"{t['id']}/{base} names its expected stage"):
                continue
            want = m.group(1)
            r = grade.grade(t, fenced(read(w), FENCE[kind]))
            _time(t, base, r)
            check(not r["passed"] and r["stage"] == want and not r["error"],
                  f"{t['id']}/{base} fails at {want} ({r['seconds']}s)",
                  f"passed={r['passed']} stage={r['stage']} {r['detail'][:600]}")


def test_a_reply_without_code_fails_at_extract():
    for t in tasks():
        kind = t["grader"]["kind"]
        r = grade.grade(t, PROSE)
        check(not r["passed"] and r["stage"] == "extract",
              f"{t['id']} prose-only reply fails at extract", r["stage"])
        r = grade.grade(t, fenced(read(grade._p(t["reference"])), WRONG_FENCE[kind]))
        check(not r["passed"] and r["stage"] == "extract",
              f"{t['id']} reference in a ```{WRONG_FENCE[kind]} fence fails at extract",
              r["stage"])


def report_timings() -> None:
    if not TIMES:
        return
    print(f"\n{'=' * 70}\n  grading time per call (seconds)")
    kinds = sorted({x["kind"] for x in TIMES})
    for k in kinds:
        for what in ("reference", "wrong"):
            xs = [x["seconds"] for x in TIMES if x["kind"] == k
                  and (x["what"] == "reference") == (what == "reference")]
            if xs:
                print(f"  {k:5s} {what:9s} n={len(xs):3d}  median {statistics.median(xs):6.2f}"
                      f"  mean {statistics.mean(xs):6.2f}  max {max(xs):6.2f}")
    worst = sorted(TIMES, key=lambda x: -x["seconds"])[:5]
    print("  slowest:", ", ".join(f"{x['id']}/{x['what']} {x['seconds']}s" for x in worst))
    print(f"  total grading time: {sum(x['seconds'] for x in TIMES):.1f}s "
          f"over {len(TIMES)} calls")
    path = os.path.join(grade.WORK, "grade_timings.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for x in TIMES:
            f.write(json.dumps(x) + "\n")
    print(f"  per-call timings: {path}")


def main() -> int:
    global ONLY, DIRS
    DIRS = "--dirs" in sys.argv
    if "--only" in sys.argv:
        ONLY = sys.argv[sys.argv.index("--only") + 1]
    t0 = time.perf_counter()
    for fn in (test_extraction_rules,
               test_toolchains_are_alive,
               test_typegpu_is_the_indexed_version,
               test_three_exports_are_alive_and_match_the_index,
               test_task_rows_are_complete,
               test_tsl_ground_truth_is_version_marked,
               test_every_reference_passes,
               test_every_alternative_correct_answer_passes,
               test_every_wrong_answer_fails_at_its_stage,
               test_a_reply_without_code_fails_at_extract):
        print(f"\n--- {fn.__name__} ---", flush=True)
        n0 = len(_results)
        try:
            fn()
        except Exception:                                        # noqa: BLE001
            check(False, f"{fn.__name__} itself raised",
                  traceback.format_exc().strip().split("\n")[-1])
        shown = 0
        for ok, name, detail in _results[n0:]:
            if not ok or shown < 400:
                print(("  pass  " if ok else "  FAIL  ") + name
                      + (f"   <- {detail}" if not ok and detail else ""), flush=True)
                shown += 1
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
