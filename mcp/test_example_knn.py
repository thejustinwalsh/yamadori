#!/usr/bin/env python
"""The package examples stage and the example kNN label index
(mcp/package_examples.py, mcp/example_knn.py). No GPU, no network, no
model: a fake GitHub raw transport (package_net.TRANSPORT), a fake held
package source tree (YAMADORI_PACKAGES_SRC), every store a temp dir, and a
deterministic embedder (hashed bag of identifiers, unit norm).

    python mcp/test_example_knn.py      -> "N/M checks passed"

  1. DISCOVERY: the linked directory first, the convention's roots, groups
     (a multi-file demo is one group, a file at a root is its own).
  2. FILTERS: node_modules, a built dir, a lockfile, the extension, minified
     by name and by content, a generated header, a duplicate, a credential,
     a bidi character, a blob that is not the tree's; zero-width typography
     stripped and kept.
  3. LABELS: each file's own imports; unheld imports (Node's own modules
     aside) are the "onboard next" list.
  4. IDEMPOTENCY: a blob already stored is not fetched again.
  5. IMPORT STRIPPING: multi-line, side-effect, require, re-export, dynamic;
     line numbers kept.
  6. THE INDEX: chunks without import lines, multi-package chunk labels, no
     example text in the rows, labelless chunks not indexed.
  7. k BY LEAVE-ONE-GROUP-OUT: a constructed case with a known curve; ties
     -> the smaller k; the held-out group is never read (instrumented).
  8. vote(): no index -> ok False; a tie -> every package at the argmax.
  9. UPKEEP: index_state / schedule idempotent, the idle flag, stale after a
     manifest changes.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import offline_stores  # noqa: E402
_TMP = offline_stores.isolate("yamadori_test_example_knn_")
SRC = os.path.join(_TMP, "_src")
os.environ["YAMADORI_PKG_REGISTRY_DIR"] = os.path.join(_TMP, "registry")
os.environ["YAMADORI_PACKAGES_SRC"] = SRC
os.environ["YAMADORI_TS_LIB"] = os.path.join(_TMP, "no_ts_lib")
os.environ["YAMADORI_NODE_TYPES"] = os.path.join(_TMP, "node_types")
os.environ["YAMADORI_PKG_DIR"] = os.path.join(_TMP, "packages")
os.environ.setdefault("YAMADORI_GPU_ROOM", "0")


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


# Two held packages with a few unique exports, one shared name, one English.
_write(os.path.join(SRC, "fakeecs@1.0.0", "index.d.ts"),
       "export declare function spawnThing(): void;\n"
       "export declare function queryThings(): void;\n"
       "export declare function sharedName(): void;\n"
       "export declare const world: number;\n")
_write(os.path.join(SRC, "fakemath@2.0.0", "index.d.ts"),
       "export declare function vecAdd(): void;\n"
       "export declare function vecScale(): void;\n"
       "export declare function sharedName(): void;\n")
_write(os.path.join(_TMP, "node_types", "fs.d.ts"),
       'declare module "fs" { export function readFileSync(): void; }\n'
       'declare module "node:fs" { export * from "fs"; }\n')

import skill_packages as SP  # noqa: E402
SP.prose_words = lambda names=None: {"world", "the", "thing"}
SP.tokenizer_words = lambda: set()
SP.REFERENCE_TYPES = ()

import example_knn as K  # noqa: E402
import jobs  # noqa: E402
import onboarding  # noqa: E402
import package_examples as PE  # noqa: E402
import package_net as net  # noqa: E402

_results: list[tuple[bool, str, str]] = []


def check(ok, name, detail=""):
    _results.append((bool(ok), name, str(detail)[:400]))


# ---------------------------------------------------------------- fixtures --
def fake_embed(texts, is_query=False):
    """A hashed bag of identifiers, unit norm: deterministic, and code that
    shares names lands near."""
    import re

    import numpy as np
    out = np.zeros((len(texts), 128), dtype=np.float32)
    for i, t in enumerate(texts):
        body = t.split("Query:", 1)[-1]
        for tok in re.findall(r"[A-Za-z_$][\w$]*", body):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            out[i, h % 128] += 1.0
        n = np.linalg.norm(out[i])
        out[i] = out[i] / n if n else out[i]
        if not n:
            out[i, 0] = 1.0
    return out


K.EMBED = fake_embed

OWNER, REPO, COMMIT = "acme", "widgets", "c0ffee" + "0" * 34
LONG = "\n".join(f"export const longConstantNumber{i} = spawnThing() + {i};"
                 for i in range(30))
PACKED = ("var a=function(){return spawnThing();};" * 40 + "\n") * 3

FILES = {
    "examples/basic/main.ts": (
        "import { spawnThing, queryThings } from 'fakeecs'\n"
        "import {\n  vecAdd,\n  vecScale,\n} from 'fakemath'\n\n"
        "export function stepWorld(dt: number) {\n"
        "  const things = queryThings()\n"
        "  const next = vecAdd(things, vecScale(dt))\n"
        "  spawnThing(next)\n  return next\n}\n"),
    "examples/basic/util.ts": (
        "import { vecAdd } from 'fakemath'\n\n"
        "export function addAll(xs: number[]) {\n"
        "  return xs.reduce((a, b) => vecAdd(a, b), 0)\n}\n"),
    "examples/single.ts": (
        "import * as ecs from 'fakeecs'\n\n"
        "export function spawnMany(n: number) {\n"
        "  for (let i = 0; i < n; i++) ecs.spawnThing(i)\n"
        "  return ecs.queryThings()\n}\n"),
    "examples/unheld.ts": (
        "import leftPad from 'left-pad-x'\nimport { readFileSync } from 'fs'\n"
        "import { spawnThing } from 'fakeecs'\n"
        "export const padded = leftPad(String(spawnThing()), 8)\n"),
    "examples/zw.ts": ("import { spawnThing } from 'fakeecs'\n"
                       "// zero​width typography in a comment\n"
                       "export const made = spawnThing()\n"),
    "examples/dup/a.ts": "import { spawnThing } from 'fakeecs'\n" + LONG,
    "examples/dup2/a.ts": "import { spawnThing } from 'fakeecs'\n" + LONG,
    "examples/dist/bundle.js": "export const x = 1\n",
    "examples/node_modules/x/index.js": "export const x = 1\n",
    "examples/big.min.js": "var a=1;\n",
    "examples/packed.js": PACKED,
    "examples/gen.ts": ("// @generated by the schema tool; do not edit\n"
                        "import { spawnThing } from 'fakeecs'\n"
                        "export const g = spawnThing()\n"),
    "examples/secret.ts": ("import { spawnThing } from 'fakeecs'\n"
                           "const key = 'sk-" + "A1b2C3d4" * 4 + "'\n"
                           "export const s = spawnThing(key)\n"),
    "examples/bidi.ts": ("import { spawnThing } from 'fakeecs'\n"
                         "const admin = 'user‮'\n"
                         "export const b = spawnThing(admin)\n"),
    "examples/mismatch.ts": "export const m = 1\n",
    "examples/readme.md": "# examples\n",
    "examples/package-lock.json": "{}\n",
    "src/index.ts": "export const notAnExample = 1\n",
    "demos/space/app.tsx": (
        "import { vecScale } from 'fakemath'\n\n"
        "export function App() {\n  return vecScale(2)\n}\n"),
    "linked/demo/one.ts": ("import { queryThings } from 'fakeecs'\n"
                           "export const q = queryThings()\n"),
}


def tree_of(files):
    entries = []
    dirs = set()
    for p, t in sorted(files.items()):
        b = t.encode("utf-8")
        sha = PE.git_blob_sha(b)
        if p == "examples/mismatch.ts":
            sha = "0" * 40
        entries.append({"path": p, "type": "blob", "sha": sha,
                        "size": len(b)})
        parts = p.split("/")
        for i in range(1, len(parts)):
            dirs.add("/".join(parts[:i]))
    for d in sorted(dirs):
        entries.append({"path": d, "type": "tree", "sha": "t", "size": None})
    return {"entries": entries, "truncated": False, "walked": []}


GETS: list[str] = []


def install_transport(files):
    by_url = {net.raw_url(OWNER, REPO, COMMIT, p): t.encode("utf-8")
              for p, t in files.items()}

    def transport(url, headers, max_bytes):
        GETS.append(url)
        body = by_url.get(url)
        if body is None:
            return net.Response(404, b"", {}, url)
        if max_bytes is not None and len(body) > max_bytes:
            raise net.TooLarge(url)
        return net.Response(200, body, {}, url)
    net.TRANSPORT = transport


PKG = {"ecosystem": "npm", "name": "fakeecs", "version": "1.0.0",
       "commit": COMMIT, "repository": {"owner": OWNER, "repo": REPO,
                                        "directory": None}}
DID = "ds_examples_1"


# ================================================================ tests ==
def test_strip_imports():
    src = ("import {\n  a,\n  b as c,\n} from 'x'\nimport 'side-effect'\n"
           "const { d } = require('y')\nrequire('z')\n"
           "export * from './r'\nexport { e } from \"q\";\n"
           "const m = await import('three/tsl')\nuseIt(a, c, d)\n")
    out = PE.strip_imports(src)
    check(out.count("\n") == src.count("\n"), "[strip] line count kept",
          (out.count("\n"), src.count("\n")))
    check(PE.is_generated("// Code generated by protoc. DO NOT EDIT.\nx")
          and PE.is_generated("/* This file was automatically generated */")
          and not PE.is_generated("// Terrain generated from simplex "
                                  "noise\nexport const t = 1")
          and not PE.is_generated("export const x = 1 // @generated"),
          "[filter] a generated HEADER drops a file; a demo describing what "
          "it generates, or the word past the header, does not")
    check("from" not in out and "require" not in out and "'x'" not in out
          and "three/tsl" not in out and "useIt(a, c, d)" in out,
          "[strip] import, side-effect, require, re-export and a dynamic "
          "import's specifier removed; code kept", out)


def test_collect():
    install_transport(FILES)
    GETS.clear()
    rec = PE.collect(DID, PKG, tree_of(FILES), linked_dirs=["linked/demo"],
                     beat=lambda s: None)
    summ = onboarding.read_json(DID, PE.SUMMARY)
    rows = PE.manifest_rows(DID)
    paths = {r["path"] for r in rows}
    roots = [r["path"] for r in summ["roots"]]
    check(roots[0] == "linked/demo" and "examples" in roots and "demos"
          in roots and "src" not in roots, "[discovery] the linked directory "
          "first, then the directories the convention names", roots)
    reasons = {d["path"]: d["reason"] for d in summ["dropped_files"]}
    want = {"examples/node_modules/x/index.js": "node_modules",
            "examples/dist/bundle.js": "built_dir",
            "examples/package-lock.json": "lockfile",
            "examples/big.min.js": "minified",
            "examples/packed.js": "minified",
            "examples/gen.ts": "generated",
            "examples/mismatch.ts": "blob_mismatch"}
    for p, r in want.items():
        check(reasons.get(p) == r, f"[filter] {p} dropped as {r}",
              reasons.get(p))
    check(str(reasons.get("examples/secret.ts", "")).startswith(
        "credential"), "[filter] a demo API key drops the file (the skill "
        "screen's credential rule)", reasons.get("examples/secret.ts"))
    check("bidi control" in str(reasons.get("examples/bidi.ts", "")),
          "[filter] a bidi character drops the file",
          reasons.get("examples/bidi.ts"))
    dup = [p for p in ("examples/dup/a.ts", "examples/dup2/a.ts")
           if reasons.get(p) == "duplicate"]
    check(len(dup) == 1 and len({"examples/dup/a.ts",
                                 "examples/dup2/a.ts"} & paths) == 1,
          "[filter] a near-duplicate is dropped once, by deps._dedupe's "
          "rule", dup)
    check(summ["dropped_extensions"].get(".md") == 1,
          "[filter] other extensions counted, not listed",
          summ["dropped_extensions"])
    zw = next(r for r in rows if r["path"] == "examples/zw.ts")
    stored = PE.read_stored(zw)
    check(stored is not None and "​" not in stored and
          summ["screen"]["typography_stripped_chars"] == 1,
          "[screen] zero-width typography stripped and recorded; the file "
          "is kept", summ["screen"])
    fb = summ["files_by_group"]
    check(fb.get("examples/basic") == 2 and fb.get("examples/single.ts")
          == 1 and fb.get("demos/space") == 1,
          "[groups] a multi-file demo is one group; a file at a root is "
          "its own", fb)
    main = next(r for r in rows if r["path"] == "examples/basic/main.ts")
    check(main["packages"] == ["fakeecs", "fakemath"] and
          main["imports"].get("vecScale") == "fakemath",
          "[labels] a file's own imports label it (multi-package)",
          (main["packages"], main["imports"]))
    single = next(r for r in rows if r["path"] == "examples/single.ts")
    check(single["imports"].get("spawnThing") == "fakeecs",
          "[labels] a namespace import's members are its bindings",
          single["imports"])
    un = next(r for r in rows if r["path"] == "examples/unheld.ts")
    check(un["unheld"] == ["left-pad-x"], "[labels] an unheld import is "
          "recorded; Node's own `fs` is not a package", un["unheld"])
    check([u["package"] for u in summ["unheld"]] == ["left-pad-x"],
          "[labels] the onboard-next list, by group count", summ["unheld"])
    check("text" not in main and main["blob"] and main["bytes"] > 0,
          "[manifest] rows carry blob, bytes and labels, not the text",
          sorted(main))
    check(rec["example_files"] == len(rows) and rec["example_groups"]
          == summ["groups"] and rec["per_package"]["fakeecs"]["files"] >= 4,
          "[counts] the stage's counts", rec)
    check(summ["bytes_fetched"] > 0 and summ["fetched"] > 0,
          "[bytes] bytes fetched recorded", summ["bytes_fetched"])
    # IDEMPOTENCY: nothing already stored is fetched again.
    GETS.clear()
    rec2 = PE.collect(DID, PKG, tree_of(FILES), linked_dirs=["linked/demo"],
                      beat=lambda s: None)
    summ2 = onboarding.read_json(DID, PE.SUMMARY)
    check(not GETS and summ2["fetched"] == 0 and rec2["example_files"]
          == rec["example_files"], "[idempotent] a blob whose sha is "
          "already stored is not fetched again (0 GETs)", (len(GETS),
                                                           summ2["fetched"]))
    check(PE.collect("ds_x", dict(PKG, commit=None), None)["skipped"],
          "[skip] no commit -> skipped, recorded")
    check(PE.collect("ds_y", dict(PKG, ecosystem="python"), None)
          .get("skipped", "").startswith("not supported"),
          "[skip] not npm -> skipped, recorded")


def test_index():
    st = K.index_state()
    check(not st["fresh"] and st["why"] == "no index file",
          "[upkeep] manifests and no index: not fresh", st)
    j1 = K.schedule()
    j2 = K.schedule()
    job = jobs.get(j1) if j1 else None
    check(j1 and j2 is None and job and job["lane"] == "gpu_a4000"   # the embedder: the A4000 (jobs.GPU_SCOPES)
         
          and (job.get("payload") or {}).get("idle") is True,
          "[upkeep] schedule enqueues ONE idle-gated gpu rebuild", (j1, j2,
                                                                     job))
    rec = K.build(beat=lambda s: None)
    check(K.index_state()["fresh"], "[upkeep] fresh after the build",
          K.index_state())
    rows = [json.loads(x) for x in open(K._path(K.ROWS), encoding="utf-8")]
    multi = [r for r in rows if r["labels"] == ["fakeecs", "fakemath"]]
    check(multi, "[index] a chunk referencing two packages' bindings "
          "carries both labels", [r["labels"] for r in rows])
    check(all("text" not in r for r in rows) and all(r["group"] and
                                                     r["path"] for r in rows),
          "[index] rows carry ids, group, labels, path -- never the text",
          sorted(rows[0]))
    check(all(r["labels"] for r in rows) and
          "no_label_chunks" in rec["dropped"],
          "[index] every indexed chunk has a label; labelless ones are "
          "counted, not indexed", rec["dropped"])
    texts = [c["text"] for c in K.collect_rows()["chunks"]]
    check(texts and not any("from '" in t or "require(" in t
                            for t in texts),
          "[index] no indexed chunk holds an import or require line",
          [t[:80] for t in texts if "from '" in t][:3])
    idx = K.load()
    check(idx and len(idx["rows"]) == rec["chunks"] and
          idx["docs"].shape[0] == rec["chunks"], "[index] load() matches "
          "the build", rec)
    check(rec["k"] is not None and rec["k_n"] > 0 and rec["curve"],
          "[k] chosen by leave-one-group-out, with its n and curve",
          {k: rec[k] for k in ("k", "k_n", "k_max", "groups")})
    # A manifest change makes the index stale.
    mp = os.path.join(onboarding.out_dir(DID), PE.MANIFEST)
    with open(mp, "rb") as f:
        body = f.read()
    with open(mp, "wb") as f:
        f.write(body + b"\n")
    st = K.index_state()
    check(not st["fresh"] and "stale" in st["why"],
          "[upkeep] a changed manifest -> stale", st)
    with open(mp, "wb") as f:
        f.write(body)
    check(K.index_state()["fresh"], "[upkeep] restored -> fresh again")


def test_vote():
    v = K.vote("export function f(dt) {\n  const next = vecAdd(queryThings()"
               ", vecScale(dt))\n  spawnThing(next)\n}\n")
    check(v["ok"] and v["groups"] and v["index"],
          "[vote] a vote from the index", v)
    top = v["top"]
    mx = max(v["tally"].values())
    check(set(top) == {p for p, n in v["tally"].items() if n == mx},
          "[vote] top is every package at the argmax", v)
    # A forced tie: k=1 and the nearest group's best chunk carries two.
    saved = K._LOADED["index"]
    idx = dict(saved, k=1)
    K._LOADED["index"] = idx
    try:
        v1 = K.vote("const next = vecAdd(queryThings(), vecScale(dt));"
                    " spawnThing(next)")
    finally:
        K._LOADED["index"] = saved
    check(v1["ok"] and len(v1["groups"]) == 1 and sorted(v1["top"]) ==
          ["fakeecs", "fakemath"], "[vote] a tie returns every tied "
          "package", v1)
    d = os.environ["YAMADORI_EXAMPLE_KNN_DIR"]
    os.environ["YAMADORI_EXAMPLE_KNN_DIR"] = os.path.join(_TMP, "empty")
    K._LOADED.update(key=None, index=None)
    try:
        v0 = K.vote("spawnThing()")
    finally:
        os.environ["YAMADORI_EXAMPLE_KNN_DIR"] = d
        K._LOADED.update(key=None, index=None)
    check(not v0["ok"] and "no example kNN index" in v0["why"],
          "[vote] no index -> ok False with why", v0)


def _unit(deg):
    import math

    import numpy as np
    r = math.radians(deg)
    return np.array([math.cos(r), math.sin(r)], dtype=np.float32)


def _constructed():
    """Five groups on a circle: A at 0, 10, 20 degrees; B at 80, 90. Each
    group's one query file sits on its group. Known curve: k=1 5/5, k=2
    5/5 (B's tie broken by the best rank), k=3 3/5 (B outvoted)."""
    import numpy as np
    degs = [0, 10, 20, 80, 90]
    labs = [("A",), ("A",), ("A",), ("B",), ("B",)]
    D = np.vstack([_unit(d) for d in degs])
    return K.Neighbours(D, list(range(5)), labs, D.copy(), list(range(5)),
                        labs, [f"g{i}" for i in range(5)])


def test_logo_known():
    nb = _constructed()
    sel = K.logo(nb, range(5))
    curve = [(c["k"], c["correct"], c["n"]) for c in sel["curve"]]
    check(curve == [(1, 5, 5), (2, 5, 5), (3, 3, 5)] and sel["k_max"] == 3,
          "[k] the constructed curve: k=1 5/5, k=2 5/5, k=3 3/5", curve)
    check(sel["k"] == 1 and sel["n"] == 5,
          "[k] ties in accuracy -> the smaller k; n is the query files",
          sel)
    t = K.tally(nb, 3, 2, {3})
    check(t["order"][0] == "B" and t["tally"] == {"B": 1, "A": 1},
          "[tally] a vote tie is broken by the best rank", t)


class Instrumented(K.Neighbours):
    ACTIVE: set = set()
    VIOLATIONS: list = []
    TOUCHED: list = []

    def ranked(self, f, exclude):
        for g in super().ranked(f, exclude):
            Instrumented.TOUCHED.append(g)
            if g in Instrumented.ACTIVE:
                Instrumented.VIOLATIONS.append(("ranked", f, g))
            yield g

    def queries_of(self, g):
        if g in Instrumented.ACTIVE:
            Instrumented.VIOLATIONS.append(("queries_of", g))
        return super().queries_of(g)


def test_logo_nested():
    base = _constructed()
    nb = Instrumented.__new__(Instrumented)
    nb.__dict__.update(base.__dict__)
    for g in range(5):
        Instrumented.ACTIVE = {g}
        Instrumented.TOUCHED = []
        sel = K.logo(nb, range(5), exclude={g})
        check(sel["k"] is not None and Instrumented.TOUCHED,
              f"[nested] inner selection for held-out g{g} ran", sel)
    check(not Instrumented.VIOLATIONS, "[nested] choosing k for a held-out "
          "group never read that group (no query of it, no vote from it)",
          Instrumented.VIOLATIONS[:5])
    Instrumented.ACTIVE = set()
    sel = K.logo(nb, range(5), exclude={3, 4})
    check(sel["k_max"] == 3 and sel["n"] == 3,
          "[nested] k's range and n are the training folds' own", sel)


def main() -> int:
    tests = [test_strip_imports, test_collect, test_index, test_vote,
             test_logo_known, test_logo_nested]
    for t in tests:
        try:
            t()
        except Exception as e:                                   # noqa: BLE001
            check(False, f"{t.__name__} raised {type(e).__name__}: {e}",
                  traceback.format_exc()[-1200:])
    bad = [r for r in _results if not r[0]]
    for ok, name, detail in _results:
        print(f"  {'ok  ' if ok else 'FAIL'} {name}"
              + ("" if ok else f"\n        {detail}"))
    print(f"\n{len(_results) - len(bad)}/{len(_results)} checks passed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
