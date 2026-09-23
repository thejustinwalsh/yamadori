#!/usr/bin/env python
"""Package indexing, asserted. No GPU, no network, no real package store.

THE DEFECT THIS GATES

Packages were indexed by running scripts/index_code.py on the tarball root,
and index_code prunes dist/ and build/ -- right for a user's repository, wrong
for a package whose only code is its built output. Measured on the real
tarballs: @react-three/fiber@10.0.0-alpha.5 indexed 9,372 of 4,683,801 bytes
(the README), @webgpu/types 4,815 of 122,628, postprocessing 8,384 of
2,526,387, wgpu-matrix 13,595 of 631,299. Each still had chunks, so each would
have counted as HELD and every search against it came back empty.

`deps.code_files` now picks a package's real code and `deps.index_health`
says whether an index is usable. Both are checked here on a fake package built
in a temp directory: a src/-less dist/ with ESM, CJS, UMD, a minified bundle,
a mean-line-length-minified bundle, a source map, parallel entry points and
esm/+cjs/ directories.

THE SECOND DEFECT (scripts/symbols.py)

A heritage clause was recorded as a definition of the parent: in
three@0.185.1, `AnalyticLightNode` had 8 definitions, one real and seven
`class X extends AnalyticLightNode`. Checked on fixture files per grammar.
"""
from __future__ import annotations

import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
SCRIPTS = os.path.join(REPO, "scripts")
sys.path.insert(0, HERE)
sys.path.insert(0, SCRIPTS)

_TMP = tempfile.mkdtemp(prefix="yamadori_test_package_index_")
os.environ["YAMADORI_PKG_DIR"] = os.path.join(_TMP, "store")
os.environ.pop("INDEX_FILES", None)
os.environ.pop("INDEX_APPEND", None)

import deps  # noqa: E402
import packages  # noqa: E402
import symbols as sym  # noqa: E402

_results: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> bool:
    _results.append((bool(ok), name, detail))
    return bool(ok)


# ------------------------------------------------------------------ fixtures
def _fns(prefix: str, n: int, indent: str = "") -> str:
    out = []
    for i in range(n):
        out += [f"{indent}function {prefix}Function{i}(argumentOne, argumentTwo) {{",
                f"{indent}  const {prefix}ResultValue{i} = argumentOne * {i} + argumentTwo;",
                f"{indent}  return {prefix}ResultValue{i};",
                f"{indent}}}"]
    return "\n".join(out) + "\n"


def _dts(prefix: str, n: int) -> str:
    return "".join(f"export declare function {prefix}Function{i}(argumentOne: number, "
                   f"argumentTwo: number): number;\n" for i in range(n))


def _write(root: str, rel: str, text: str) -> str:
    p = os.path.join(root, *rel.split("/"))
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    return p


CORE = _fns("core", 60)
ESM_TAIL = "export { coreFunction0, coreFunction1 };\n"


def build_dist_only(root: str) -> None:
    """A package that ships no src/: everything is built output."""
    _write(root, "package.json", '{"name": "fake-dist", "version": "1.0.0"}\n')
    _write(root, "README.md", "# fake-dist\n\nA fixture package with only built output.\n")
    # The main entry, ESM.
    _write(root, "dist/index.mjs", "import { Thing } from 'three';\n" + CORE + ESM_TAIL)
    # A second entry point that is a copy minus one line (fiber's legacy.mjs:
    # 0.9937 similar, and slightly smaller, so the main entry is the one kept).
    _write(root, "dist/legacy.mjs", "import { Thing } from 'three';\n" + CORE)
    # A DISTINCT entry point sharing the core (fiber's webgpu/index.mjs shape).
    _write(root, "dist/webgpu/index.mjs", "import { Thing } from 'three/webgpu';\n"
           + CORE + _fns("gpu", 30) + ESM_TAIL)
    # The same code as CommonJS and as a UMD bundle.
    _write(root, "dist/index.cjs", '"use strict";\nconst three = require("three");\n'
           + CORE + "module.exports = { coreFunction0 };\n")
    _write(root, "dist/umd.js",
           "(function (global, factory) {\n"
           "  typeof exports === 'object' && typeof module !== 'undefined' ? "
           "module.exports = factory() : global.Fake = factory();\n"
           "})(this, function () {\n" + _fns("core", 60, "  ") + "});\n")
    # Minified by name, minified by shape, and a source map.
    _write(root, "dist/index.min.js", "function a(b,c){return b*c}" * 50 + "\n")
    _write(root, "dist/bundle.js", ("var q=function(a,b){return a+b};" * 400 + "\n") * 3)
    _write(root, "dist/index.mjs.map", '{"version":3,"sources":["../src/index.ts"]}\n')
    # Declarations: main, an identical copy, and a distinct entry.
    _write(root, "dist/index.d.ts", _dts("core", 60))
    _write(root, "dist/legacy.d.ts", _dts("core", 60))
    _write(root, "dist/webgpu/index.d.ts", _dts("core", 60) + _dts("gpu", 30))
    # esm/ and cjs/ directories holding the same helper module.
    util = _fns("util", 25)
    _write(root, "esm/util.js", util + "export { utilFunction0 };\n")
    _write(root, "cjs/util.js", '"use strict";\n' + util
           + "exports.utilFunction0 = utilFunction0;\n")


DIST_EXPECTED = {"package.json", "README.md", "dist/index.mjs",
                 "dist/webgpu/index.mjs", "dist/index.d.ts",
                 "dist/webgpu/index.d.ts", "esm/util.js"}


def build_with_src(root: str) -> None:
    """A package that ships its source: src/ wins, build output is skipped."""
    _write(root, "package.json", '{"name": "fake-src", "version": "1.0.0"}\n')
    _write(root, "src/index.ts", "export function realSource(a: number): number {\n"
           "  return a * 2;\n}\n")
    _write(root, "src/lib/helper.ts", "export function helperInSrcLib(): number {\n"
           "  return 1;\n}\n")
    _write(root, "src/vendor.min.js", "function a(b){return b}" * 80 + "\n")
    _write(root, "dist/index.js", CORE)
    _write(root, "lib/index.js", CORE)
    _write(root, "build/index.js", CORE)
    _write(root, "types/index.d.ts", _dts("core", 3))


SRC_EXPECTED = {"package.json", "src/index.ts", "src/lib/helper.ts",
                "types/index.d.ts"}


def _rel(root: str, paths: list[str]) -> set[str]:
    return {os.path.relpath(p, root).replace("\\", "/") for p in paths}


def _pkg(name: str, builder) -> str:
    root = os.path.join(_TMP, "pkgs", name)
    if not os.path.isdir(root):
        builder(root)
    return root


# ------------------------------------------------------------------ tests
def test_the_fixture_is_not_the_real_store():
    check(os.path.abspath(deps.STORE).startswith(os.path.abspath(_TMP)),
          "deps.STORE points into the temp directory", deps.STORE)


def test_dist_only_package_selects_the_real_code():
    root = _pkg("fake-dist", build_dist_only)
    got = _rel(root, deps.code_files(root))
    check(got == DIST_EXPECTED, "a dist-only package indexes its built code, once",
          f"extra={sorted(got - DIST_EXPECTED)} missing={sorted(DIST_EXPECTED - got)}")
    check("dist/index.min.js" not in got, ".min.js is skipped")
    check("dist/bundle.js" not in got,
          "a bundle minified by shape (mean line > MINIFIED_MEAN_LINE) is skipped")
    check("dist/index.mjs.map" not in got, "source maps are skipped")
    check("dist/index.cjs" not in got and "dist/umd.js" not in got,
          "the CJS and UMD copies of the ESM code are skipped")
    check("cjs/util.js" not in got and "esm/util.js" in got,
          "esm/ is indexed and its cjs/ duplicate is not")
    check("dist/legacy.mjs" not in got and "dist/legacy.d.ts" not in got,
          "a near-identical same-format copy is skipped", sorted(got))
    check("dist/webgpu/index.mjs" in got and "dist/index.mjs" in got,
          "a distinct entry point sharing the core is KEPT beside the main one",
          sorted(got))


def test_minified_detection_uses_the_mean_not_the_max():
    root = os.path.join(_TMP, "minified")
    # postprocessing's build/index.js: a 66,812-char line among ~17k lines.
    readable = _write(root, "readable.js",
                      _fns("big", 600) + "const shader = '" + "x" * 70000 + "';\n")
    check(not deps.is_minified(readable),
          "one 70k-char line (an inlined shader) does not make a readable file minified")
    check(deps.is_minified(_write(root, "packed.js", "var a=1;" * 5000)),
          "a single 40k-char line file is minified")
    # SMAAPass.js shape: a real class plus a base64 texture on one line.
    smaa = _write(root, "SMAAPass.js",
                  "class SMAAPass extends Pass {\n  constructor() { super(); }\n}\n"
                  "const areaImage = 'data:image/png;base64," + "iVBORw0KGgoAAAANSUhEUg" * 400
                  + "';\n")
    check(not deps.is_minified(smaa),
          "a source file whose long line is embedded DATA is not minified")
    check(deps.is_minified(_write(root, "x.min.mjs", "export const a = 1;\n")),
          "`.min.` in the name is minified whatever the content")


def test_module_format_is_not_fooled_by_an_iife_body():
    text = ("var LIB = (() => {\n  var x = [\n        import_three10.RepeatWrapping,\n"
            "  ];\n})();\nif(typeof module===\"object\")module.exports=LIB;\n")
    check(deps.module_format("/p/build/lib.js", text) == "cjs",
          "an indented `import_three10.X` inside a UMD bundle is not ESM",
          deps.module_format("/p/build/lib.js", text))
    check(deps.module_format("/p/dist/a.js", "import { x } from 'y';\n") == "esm",
          "a column-0 import is ESM")
    check(deps.module_format("/p/dist/a.cjs", "export const a = 1;\n") == "cjs",
          "the .cjs extension decides before content")


def test_src_package_prefers_src():
    root = _pkg("fake-src", build_with_src)
    check(deps.has_src(root), "a src/ with code counts as source")
    got = _rel(root, deps.code_files(root))
    check(got == SRC_EXPECTED, "src/ is indexed; dist/, build/ and top-level lib/ are not",
          f"extra={sorted(got - SRC_EXPECTED)} missing={sorted(SRC_EXPECTED - got)}")


def test_repository_walk_is_unchanged():
    import index_code
    root = _pkg("fake-dist", build_dist_only)
    walked = _rel(root, list(index_code.iter_files(root)))
    check(not any(p.startswith("dist/") for p in walked),
          "index_code.iter_files still prunes dist/ for a repository", sorted(walked))
    check(walked == {"package.json", "README.md", "esm/util.js", "cjs/util.js"},
          "and walks exactly what it walked before", sorted(walked))


def _old_style_index(root: str, db: str) -> None:
    env = dict(os.environ, CODE_INDEX_DB=db, INDEX_NO_EMBED="1")
    env.pop("INDEX_FILES", None)
    subprocess.run([sys.executable, os.path.join(SCRIPTS, "index_code.py"), root],
                   env=env, capture_output=True, text=True, timeout=300, check=True)


def test_health_separates_readme_only_from_real():
    root = os.path.join(_TMP, "pkgs", "readme-only")
    _write(root, "README.md", "# readme only\n\n" + "Some prose about the API. " * 20 + "\n")
    _write(root, "package.json", '{"name": "readme-only"}\n')
    _write(root, "dist/index.mjs", CORE)
    db = os.path.join(_TMP, "old-readme-only.sqlite3")
    _old_style_index(root, db)
    h = deps.index_health(db)
    check(h["chunks"] > 0, "the old walk DOES write chunks (the README)", str(h["chunks"]))
    check(h["defs"] == 0 and h["code_chunks"] == 0,
          "but no definitions and no code chunks", f"defs={h['defs']} code={h['code_chunks']}")
    check(h["ok"] is False, "so index_health says it is not usable")
    check(h["bytes_indexed"] < h["bytes_fetched"] and h["ratio"] < 0.5,
          "and the byte ratio shows how little was read", str(h["ratio"]))
    missing = deps.index_health(os.path.join(_TMP, "nope.sqlite3"))
    check(missing["exists"] is False and missing["ok"] is False,
          "a missing database is not ok")


def test_index_package_end_to_end():
    root = _pkg("fake-dist", build_dist_only)
    db = os.path.join(_TMP, "store", "fake-dist@1.0.0.sqlite3")
    h = deps.index_package("fake-dist", "1.0.0", embed=False, db=db, src=root)
    check(h.get("installed") is True and os.path.exists(db),
          "a healthy index is installed", str(h.get("error")))
    check(not os.path.exists(db + ".building"), "no build file is left behind")
    check(h["ok"] and h["defs"] >= 150,
          "the symbol table holds the bundle's definitions", f"defs={h['defs']}")
    expect = sum(os.path.getsize(os.path.join(root, *p.split("/"))) for p in DIST_EXPECTED)
    check(h["bytes_selected"] == expect, "bytes_selected is exactly the selected files",
          f"{h['bytes_selected']} vs {expect}")
    fetched = deps._tree_bytes(root)
    check(h["bytes_fetched"] == fetched, "bytes_fetched is the whole package",
          f"{h['bytes_fetched']} vs {fetched}")
    check(h["ratio"] == round(h["bytes_indexed"] / fetched, 4),
          "ratio = bytes_indexed / bytes_fetched", str(h["ratio"]))
    check(h["zero_vectors"] == h["chunks"] and h["vectors"] == "none",
          "without --embed every vector is zero, and health says so",
          f"{h['zero_vectors']}/{h['chunks']} {h['vectors']}")
    con = sqlite3.connect(db)
    try:
        paths = {r[0] for r in con.execute("SELECT DISTINCT path FROM chunks")}
        listed = {r[0] for r in con.execute("SELECT path FROM package_files")}
        where = con.execute("SELECT path FROM defs WHERE name='gpuFunction3'").fetchall()
        roots = [r[0] for r in con.execute("SELECT path FROM roots")]
    finally:
        con.close()
    check("dist/index.mjs" in paths and "dist/webgpu/index.mjs" in paths,
          "stored paths keep their dist/ prefix", sorted(paths))
    check(listed == DIST_EXPECTED, "the selection is recorded in package_files")
    check(sorted(where) == [("dist/webgpu/index.d.ts",), ("dist/webgpu/index.mjs",)],
          "a definition only in the webgpu entry is found there", str(where))
    check(roots == [os.path.abspath(root)], "the root is the package directory", str(roots))

    # A symbol-only rebuild must re-read the same files, not re-walk with
    # repo rules and strip every definition that lives in dist/.
    import reindex_symbols
    before = h["defs"]
    r = reindex_symbols.rebuild(db)
    check(r.get("defs_after") == before,
          "reindex_symbols keeps the dist/ definitions", str(r))


def test_unhealthy_index_is_not_installed():
    root = os.path.join(_TMP, "pkgs", "only-minified")
    _write(root, "README.md", "# only minified\n\n" + "Prose. " * 40 + "\n")
    _write(root, "dist/index.min.js", "function a(b,c){return b*c}" * 50 + "\n")
    db = os.path.join(_TMP, "store", "only-minified@1.0.0.sqlite3")
    h = deps.index_package("only-minified", "1.0.0", db=db, src=root)
    check(h.get("installed") is False and not os.path.exists(db),
          "an index with no definitions is refused, not installed", str(h.get("error")))
    check(not os.path.exists(db + ".building"), "and its build file is removed")


def test_ensure_for_attempts_once_and_rebuilds_unusable():
    calls: list[tuple[str, str]] = []
    real_plan, real_index = deps.plan, deps.index_package
    deps.plan = lambda root: [{"name": "fake-dist", "version": "1.0.0", "imports": 3,
                               "chunks": 0},
                              {"name": "readme-pkg", "version": "2.0.0", "imports": 3,
                               "chunks": 5}]
    deps.index_package = lambda n, v, embed=False: calls.append((n, v)) or {}
    # readme-pkg HAS chunks but no definitions: the old gate skipped it.
    readme_db = deps.db_path("readme-pkg", "2.0.0")
    shutil.copy(os.path.join(_TMP, "old-readme-only.sqlite3"), readme_db)
    packages._ATTEMPTED.clear()
    try:
        packages.ensure_for(_TMP)
        packages.ensure_for(_TMP)
        deadline = time.time() + 10
        while len(calls) < 1 and time.time() < deadline:
            time.sleep(0.05)
        time.sleep(0.2)
    finally:
        deps.plan, deps.index_package = real_plan, real_index
    check(("readme-pkg", "2.0.0") in calls,
          "a package with chunks but no definitions is re-indexed", str(calls))
    check(("fake-dist", "1.0.0") not in calls,
          "a package whose index is healthy is left alone", str(calls))
    check(len(calls) == 1, "two requests start one attempt, not two", str(calls))


def test_spec_parsing():
    check(deps._split_spec("@react-three/fiber@10.0.0-alpha.5")
          == ("@react-three/fiber", "10.0.0-alpha.5"), "a scoped name@version splits")
    check(deps._split_spec("three@0.186.0") == ("three", "0.186.0"),
          "an unscoped name@version splits")


# --------------------------------------------------------------- heritage
HERITAGE = {
    "base.js": "export class AnalyticLightNode {\n  setup() { return 1; }\n}\n",
    "a.js": "import { AnalyticLightNode } from './base.js';\n"
            "export class PointLightNode extends AnalyticLightNode { run() {} }\n"
            "const Expr = class extends OtherBase {};\n",
    "b.ts": "export class Child extends BaseTs implements IFoo, IBar { run(): void {} }\n"
            "export interface Ext extends ParentIface { x: number }\n",
    "c.java": "public class ChildJ extends BaseJ implements IFooJ { void run() {} }\n",
    "d.cpp": "class ChildC : public BaseC, private OtherC { void run(); };\n",
    "e.rs": "trait Shape: Display { fn area(&self) -> f64; }\n"
            "impl Display for Circle { fn fmt(&self) {} }\n"
            "fn f<T: Clone>(t: T) {}\n",
}


def test_heritage_is_a_reference_not_a_definition():
    root = os.path.join(_TMP, "heritage")
    defs: dict[str, list] = {}
    refs: set[str] = set()
    for fn, text in HERITAGE.items():
        p = _write(root, fn, text)
        d, r = sym.extract(p, fn)
        for row in d:
            defs.setdefault(row[0], []).append((fn, row[1]))
        refs |= {row[0] for row in r}
    check(defs.get("AnalyticLightNode") == [("base.js", "class")],
          "a class extended elsewhere has exactly one definition, its own",
          str(defs.get("AnalyticLightNode")))
    check("PointLightNode" in defs and "AnalyticLightNode" in refs,
          "the subclass is defined, and the parent is referenced from it")
    for parent in ("OtherBase", "BaseTs", "IFoo", "IBar", "ParentIface", "BaseJ",
                   "IFooJ", "BaseC", "OtherC", "Clone"):
        check(parent not in defs, f"heritage `{parent}` is not a definition",
              str(defs.get(parent)))
        check(parent in refs, f"heritage `{parent}` is a reference")
    check(("e.rs", "impl") in defs.get("Circle", []) and "Display" not in defs,
          "`impl Display for Circle` names Circle, not the trait",
          f"Circle={defs.get('Circle')} Display={defs.get('Display')}")
    for real in ("Child", "Ext", "ChildJ", "ChildC", "Shape"):
        check(real in defs, f"`{real}` is still defined")


def main() -> int:
    for fn in (test_the_fixture_is_not_the_real_store,
               test_dist_only_package_selects_the_real_code,
               test_minified_detection_uses_the_mean_not_the_max,
               test_module_format_is_not_fooled_by_an_iife_body,
               test_src_package_prefers_src,
               test_repository_walk_is_unchanged,
               test_health_separates_readme_only_from_real,
               test_index_package_end_to_end,
               test_unhealthy_index_is_not_installed,
               test_ensure_for_attempts_once_and_rebuilds_unusable,
               test_spec_parsing,
               test_heritage_is_a_reference_not_a_definition):
        print(f"\n--- {fn.__name__} ---")
        n0 = len(_results)
        try:
            fn()
        except Exception:                                        # noqa: BLE001
            check(False, f"{fn.__name__} itself raised",
                  traceback.format_exc().strip().split("\n")[-1])
        for ok, name, detail in _results[n0:]:
            print(("  pass  " if ok else "  FAIL  ") + name
                  + (f"   <- {detail}" if not ok and detail else ""))

    passed = sum(1 for ok, _, _ in _results if ok)
    total = len(_results)
    print(f"\n{'=' * 70}\n  {passed}/{total} checks passed")
    print(f"  temp directory: {_TMP}")
    if passed == total:
        shutil.rmtree(_TMP, ignore_errors=True)
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
