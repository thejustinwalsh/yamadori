#!/usr/bin/env python
"""Index the open-source packages a repository actually imports.

WHY THIS IS THE GOOD USE OF SERVER-SIDE FETCHING

Fetching the user's own repository to the server needs credentials, stores a
copy of private source, and is redundant when the client can run the tools
locally. Fetching its DEPENDENCIES has none of those problems: they are public,
they are pinned to exact versions by a lockfile, and every user on the same
version wants the same index. `three@0.185.1` is indexed once and serves
everyone who depends on it, forever.

It also covers the gap the local index cannot: a model asked why a TSL node
behaves a certain way has never been able to read three's source, only the
user's calls into it.

WHAT GETS INDEXED, AND WHY SO LITTLE

Measured on one real repository: 401 packages in the lockfile, 52 imported
anywhere in the source, 23 both. Indexing the lockfile would be mostly
transitive build tooling nobody asks questions about. The filter is the
owner's: index it only if this repository actually imports it.

The top entries on that repo were three@0.185.1 (134 imports) and
typegpu@0.12.5 (37), which are precisely the libraries reported as hardest for
models to get right -- so the filter selects for value on its own.

VERSION IS PART OF THE IDENTITY. An index of three@0.186 answering a question
about three@0.180 is worse than no index, because it is confidently wrong about
an API that moved. Nothing is shared between versions.
"""
from __future__ import annotations

import io
import json
import os
import re
import sqlite3
import tarfile
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
STORE = os.environ.get("YAMADORI_PKG_DIR",
                       os.path.join(HERE, "..", "index", "packages"))
SRC_CACHE = os.path.join(STORE, "_src")
REGISTRY = os.environ.get("NPM_REGISTRY", "https://registry.npmjs.org")

# Bare specifiers only. A relative path is the repo's own code and node: is the
# runtime, neither of which is a package to fetch.
_IMPORT = re.compile(
    r"""(?:from\s+|require\(\s*|import\s*\(\s*)['"]([^'"./][^'"]*)['"]""")

# pnpm and npm both write `name@version` keys; the leading indent distinguishes
# a resolved entry from a range in the importers block.
_LOCK = re.compile(r"^  (@?[a-z0-9][\w.\-]*(?:/[\w.\-]+)?)@([0-9][^\s:(]*)", re.M)

SOURCE_EXTS = {".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".d.ts",
               ".wgsl", ".glsl", ".rs", ".md"}


def package_of(specifier: str) -> str:
    """`three/examples/jsm/x` -> `three`; `@scope/pkg/deep` -> `@scope/pkg`."""
    parts = specifier.split("/")
    return "/".join(parts[:2]) if specifier.startswith("@") else parts[0]


def imported(root: str, limit_files: int = 4000) -> dict[str, int]:
    """Packages this repository actually imports, with how often."""
    counts: dict[str, int] = {}
    seen = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames
                       if d not in {"node_modules", ".git", "dist", "build",
                                    "target", ".next", "coverage"}]
        for fn in filenames:
            if os.path.splitext(fn)[1].lower() not in {".ts", ".tsx", ".js",
                                                       ".jsx", ".mjs", ".cjs"}:
                continue
            seen += 1
            if seen > limit_files:
                return counts
            try:
                with open(os.path.join(dirpath, fn), encoding="utf-8",
                          errors="ignore") as f:
                    src = f.read()
            except OSError:
                continue
            for spec in _IMPORT.findall(src):
                if spec.startswith("node:"):
                    continue
                p = package_of(spec)
                counts[p] = counts.get(p, 0) + 1
    return counts


def locked(root: str) -> dict[str, str]:
    """Exact versions from whichever lockfile is present.

    The manifest cannot be used: pnpm catalogs, workspace protocols and ranges
    like '>=0.185.0 <0.186' are all unresolvable without it.
    """
    for name in ("pnpm-lock.yaml", "package-lock.json", "yarn.lock"):
        p = os.path.join(root, name)
        if not os.path.exists(p):
            continue
        try:
            with open(p, encoding="utf-8", errors="ignore") as f:
                txt = f.read()
        except OSError:
            continue
        if name == "package-lock.json":
            try:
                data = json.loads(txt)
            except json.JSONDecodeError:
                continue
            out = {}
            for key, meta in (data.get("packages") or {}).items():
                if not key.startswith("node_modules/"):
                    continue
                pkg = key.split("node_modules/")[-1]
                if meta.get("version"):
                    out[pkg] = meta["version"]
            if out:
                return out
        else:
            found = dict(sorted(set(_LOCK.findall(txt))))
            if found:
                return found
    return {}


def worth_indexing(root: str, min_imports: int = 2,
                   max_packages: int = 25) -> list[tuple[str, str, int]]:
    """(name, version, import_count) for the packages worth the disk."""
    use = imported(root)
    ver = locked(root)
    rows = [(n, ver[n], c) for n, c in use.items()
            if n in ver and c >= min_imports]
    rows.sort(key=lambda r: -r[2])
    return rows[:max_packages]


# ------------------------------------------------------------------ fetching
def slug(name: str, version: str) -> str:
    return f"{name.replace('/', '__').lstrip('@')}@{version}"


def db_path(name: str, version: str) -> str:
    os.makedirs(STORE, exist_ok=True)
    return os.path.join(STORE, f"{slug(name, version)}.sqlite3")


def is_indexed(name: str, version: str) -> int:
    p = db_path(name, version)
    if not os.path.exists(p):
        return 0
    try:
        con = sqlite3.connect(p)
        n = con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        con.close()
        return n
    except sqlite3.Error:
        return 0


def fetch(name: str, version: str) -> str | None:
    """Download and unpack the published tarball. Returns the source dir.

    The registry tarball, not a git clone: it is what actually gets installed,
    it carries the .d.ts files that answer most API questions, and it needs no
    credentials or clone of a history nobody will read.
    """
    dest = os.path.join(SRC_CACHE, slug(name, version))
    if os.path.isdir(dest) and os.listdir(dest):
        return dest
    short = name.split("/")[-1]
    url = f"{REGISTRY}/{name}/-/{short}-{version}.tgz"
    try:
        with urllib.request.urlopen(url, timeout=180) as r:
            blob = r.read()
    except Exception:                                            # noqa: BLE001
        return None
    os.makedirs(dest, exist_ok=True)
    try:
        with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tf:
            for m in tf.getmembers():
                if not m.isfile():
                    continue
                # Tarballs are rooted at package/; strip it, and refuse any
                # member that would escape the destination.
                rel = m.name.split("/", 1)[-1]
                if not rel or os.path.isabs(rel) or ".." in rel.split("/"):
                    continue
                if os.path.splitext(rel)[1].lower() not in SOURCE_EXTS:
                    continue
                if m.size > 2_000_000:
                    continue
                out = os.path.join(dest, rel)
                os.makedirs(os.path.dirname(out), exist_ok=True)
                src = tf.extractfile(m)
                if src:
                    with open(out, "wb") as f:
                        f.write(src.read())
    except (tarfile.TarError, OSError):
        return None
    return dest if os.listdir(dest) else None


def plan(root: str) -> list[dict]:
    """What would be fetched and indexed for this repository, and what exists."""
    out = []
    for name, version, count in worth_indexing(root):
        out.append({"name": name, "version": version, "imports": count,
                    "chunks": is_indexed(name, version)})
    return out


# ------------------------------------------------------ what a package IS
#
# THE DEFECT THIS REPLACES. Packages were indexed by running index_code.py on
# the tarball root, and index_code prunes dist/ and build/ -- right for a
# user's repository, where those are their own build output, and wrong for a
# package, where they are frequently the only code shipped. Measured on the
# real tarballs (deps.fetch, 2026-09-22):
#
#     @react-three/fiber@10.0.0-alpha.5    walked   9,372 of 4,683,801 bytes
#     @webgpu/types@0.1.74                 walked   4,815 of   122,628
#     postprocessing@6.39.5                walked   8,384 of 2,526,387
#     wgpu-matrix@3.4.2                    walked  13,595 of   631,299
#
# i.e. the README. Each would have counted as HELD (chunks > 0), so the domain
# gate offered the tools and every search came back empty -- present, not
# working (PROTOCOL rule 1).
#
# THE RULE NOW. A package that ships src/ is indexed from src/ (and anything
# else outside its build folders), which is what three, @types/three and
# three-mesh-bvh look like and what the old walk already got right. A package
# that does not is indexed from its built output, minus:
#
#   minified bundles   `.min.` in the name, or a mean line length over
#                      MINIFIED_MEAN_LINE. Max line length does NOT work:
#                      postprocessing's readable build/index.js has a 66,812
#                      char line (an inlined shader) at a mean of 36.
#   source maps        never useful as text.
#   duplicate formats  the same bundle as ESM, CJS and UMD, or the same code
#                      under two entry points. One copy is kept, ESM first.
#                      Decided by CONTENT, not by name -- koota ships its
#                      react entry twice (dist/react.js, react/index.js) and
#                      postprocessing's UMD bundle has an unrelated name.

# Code extensions that make a src/ directory count as real source.
_SRC_CODE_EXTS = {".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".rs",
                  ".wgsl", ".glsl"}
# Top-level folders that hold build output when a src/ exists beside them.
BUILT_DIRS = {"dist", "build", "lib", "esm", "cjs", "es", "umd"}
_JS_EXTS = {".js", ".mjs", ".cjs", ".jsx"}
_DOC_EXTS = {".md", ".txt", ".json", ".yaml", ".yml", ".toml"}
MAX_FILE_BYTES = 1_500_000          # the same cap index_code.iter_files applies

# Measured 2026-09-22 over the 20 JS bundles in fiber 10a5, drei 11a7,
# postprocessing 6.39.5, wgpu-matrix 3.4.2 and koota 0.6.6: every readable
# bundle has a mean line length of 12-43 chars; the three minified ones are
# 345, 19,509 and 19,615. 200 sits far from both sides.
MINIFIED_MEAN_LINE = 200
# ...AND the long lines must look like code. Count of `;{}()=` per 1,000
# chars in lines over 500 chars, measured 2026-09-22: minified code 59.9-191
# (ktx-parse, basis_transcoder, draco, chevrotain.min, lil-gui.min,
# postprocessing.min, wgpu-matrix.min); embedded data 0.0-2.4 (SMAAPass,
# RectAreaLightTexturesLib, mikktspace and zstddec wasm blobs). The first
# three.js re-index without this dropped SMAAPass.js and lost its class.
MINIFIED_CODE_PUNCT_PER_KB = 20

# Two rules, because two different things get shipped twice. Similarity is
# over distinct lines (stripped, over 20 chars). Measured 2026-09-22 on every
# same-extension pair in fiber 10a5, drei 11a7, postprocessing 6.39.5,
# wgpu-matrix 3.4.2 and koota 0.6.6:
#
# SAME FORMAT, the same file twice: |A & B| / max(|A|, |B|) against ONE kept
#   file. Real copies score 0.954-1.000 (fiber legacy.mjs/index.mjs 0.9937,
#   index.d.ts/legacy.d.ts 0.9834, koota react.js twice 0.9938). Distinct entry
#   points that share a core top out at 0.9273 (fiber index.mjs vs
#   webgpu/index.mjs) and 0.9016 (drei native vs webgpu), and MUST be kept:
#   the webgpu bundle renames `Canvas` to `Canvas$1`, so dropping index.mjs
#   because its lines are "covered" loses the definition of Canvas.
# OTHER FORMAT, the same code as CJS/UMD beside an ESM copy: the fraction of
#   the file's lines already in the kept ESM files. The format rewrite touches
#   every import/export line, so copies score lower: 0.885 (postprocessing
#   index.cjs and its UMD bundle vs index.js) to 0.987 (wgpu-matrix UMD).
DUP_SAME_FORMAT = 0.95
DUP_OTHER_FORMAT = 0.80
# Below this many distinct lines a file is kept whatever it overlaps: a
# one-line re-export covering nothing new is cheap and the ratio is noise.
DUP_MIN_LINES = 20

# Column 0 only: a UMD/IIFE bundle indents its body, and `import_three10.X`
# inside one matched a looser `\s*import\s*\w` and made it look like ESM.
_ESM = re.compile(r"^(?:export[\s{*]|import[\s{*'\"])", re.M)
_CJS = re.compile(r"\brequire\(|\bmodule\.exports\b|^\s*exports\.", re.M)


def _read(path: str) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return ""


def is_minified(path: str, text: str | None = None) -> bool:
    base = os.path.basename(path).lower()
    if ".min." in base:
        return True
    if os.path.splitext(base)[1] not in _JS_EXTS:
        return False
    text = _read(path) if text is None else text
    if not text:
        return False
    if len(text) / (text.count("\n") + 1) <= MINIFIED_MEAN_LINE:
        return False
    # Long lines can be DATA rather than packed code: SMAAPass.js carries two
    # base64 textures (mean line 219), RectAreaLightTexturesLib.js a float
    # table (mean 2,478). Both are real source with real classes. Packed code
    # is dense in code punctuation; data is not.
    longs = [ln for ln in text.splitlines() if len(ln) > 500]
    size = sum(len(ln) for ln in longs)
    if not size:
        return True
    punct = sum(ln.count(c) for ln in longs for c in ";{}()=")
    return punct * 1000 / size >= MINIFIED_CODE_PUNCT_PER_KB


def has_src(pkg_dir: str) -> bool:
    """Does this package ship its own source in src/?"""
    src = os.path.join(pkg_dir, "src")
    if not os.path.isdir(src):
        return False
    for _d, _ds, fs in os.walk(src):
        if any(os.path.splitext(f)[1].lower() in _SRC_CODE_EXTS for f in fs):
            return True
    return False


def module_format(path: str, text: str) -> str:
    """'esm', 'script' (UMD/IIFE/unknown) or 'cjs'. Ranked in that order."""
    low = path.replace("\\", "/").lower()
    base = os.path.basename(low)
    if (base.endswith(".cjs") or ".cjs." in base or "/cjs/" in low):
        return "cjs"
    if base.endswith(".mjs") or ".esm." in base or "/esm/" in low:
        return "esm"
    if _ESM.search(text):
        return "esm"
    if _CJS.search(text):
        return "cjs"
    return "script"


_RANK = {"esm": 0, "script": 1, "cjs": 2}


def _distinct_lines(text: str) -> set[int]:
    return {hash(s) for s in (ln.strip() for ln in text.splitlines())
            if len(s) > 20}


def _dedupe(paths: list[str], texts: dict[str, str],
            ranked: bool) -> list[str]:
    """Keep one copy of each bundle: best format first, then biggest first.

    Biggest first so that when two same-format files are copies, the one
    kept is the fuller. Distinct entry points are never merged -- see
    DUP_SAME_FORMAT for why line overlap alone cannot be trusted with that.
    """
    info = []
    for p in paths:
        fmt = module_format(p, texts[p]) if ranked else "esm"
        info.append((_RANK[fmt], p, _distinct_lines(texts[p])))
    info.sort(key=lambda r: (r[0], -len(r[2]), r[1]))
    seen: set[int] = set()
    kept_sets: list[set[int]] = []
    best_rank = None
    kept = []
    for rank, p, lines in info:
        if len(lines) >= DUP_MIN_LINES:
            if best_rank is not None and rank > best_rank:
                if len(lines & seen) / len(lines) >= DUP_OTHER_FORMAT:
                    continue
            elif any(len(lines & k) / max(len(lines), len(k)) >= DUP_SAME_FORMAT
                     for k in kept_sets):
                continue
        kept.append(p)
        kept_sets.append(lines)
        seen |= lines
        if best_rank is None:
            best_rank = rank
    return kept


def code_files(pkg_dir: str) -> list[str]:
    """Every file a PACKAGE index should read, as absolute paths.

    The one selection every package-indexing path uses (index_package, and
    through it packages.ensure_for). Repository indexing does not come here
    and still walks with index_code.SKIP_DIRS.
    """
    sys_path_scripts()
    import index_code as ic

    pkg_dir = os.path.abspath(pkg_dir)
    src_mode = has_src(pkg_dir)
    always_skip = set(ic.SKIP_DIRS) - {"dist", "build"}
    picked: list[str] = []
    for dirpath, dirnames, filenames in os.walk(pkg_dir):
        top = os.path.normcase(dirpath) == os.path.normcase(pkg_dir)
        dirnames[:] = sorted(
            d for d in dirnames
            if not d.startswith(".") and d not in always_skip
            and not (src_mode and (d in {"dist", "build"}
                                   or (top and d in BUILT_DIRS))))
        for fn in sorted(filenames):
            low = fn.lower()
            if low.endswith(".map") or os.path.splitext(low)[1] not in ic.EXTS:
                continue
            p = os.path.join(dirpath, fn)
            try:
                if os.path.getsize(p) > MAX_FILE_BYTES:
                    continue
            except OSError:
                continue
            picked.append(p)

    texts = {p: _read(p) for p in picked
             if os.path.splitext(p)[1].lower() in _JS_EXTS
             or p.lower().endswith((".d.ts", ".d.mts", ".d.cts"))}
    picked = [p for p in picked if not is_minified(p, texts.get(p))]
    if src_mode:
        return picked

    js = [p for p in picked if os.path.splitext(p)[1].lower() in _JS_EXTS]
    dts = [p for p in picked if p.lower().endswith((".d.ts", ".d.mts", ".d.cts"))]
    keep = set(_dedupe(js, texts, ranked=True)) | set(_dedupe(dts, texts, ranked=False))
    return [p for p in picked if p in keep or (p not in texts)]


def sys_path_scripts() -> None:
    import sys
    scripts = os.path.normpath(os.path.join(HERE, "..", "scripts"))
    if scripts not in sys.path:
        sys.path.insert(0, scripts)


# ------------------------------------------------------ is the index usable
#
# "HELD" MUST MEAN USABLE. domains.held_sources() counts a database with any
# chunk at all, and a README-only index has chunks. What separates an index of
# a package's code from an index of its README is whether the symbol table has
# anything in it: README.md, LICENSE.md and package.json yield no definitions.
# See MIN_DEFS for the numbers.

# Measured 2026-09-22 (bench numbers in the report accompanying this change):
# the four dist-only packages indexed the old way hold 0 definitions each;
# the smallest correctly indexed package, @webgpu/types, holds hundreds. The
# threshold is the floor that separates those, not a quality bar.
MIN_DEFS = 1


def _scalar(con: sqlite3.Connection, sql: str, default=0):
    try:
        row = con.execute(sql).fetchone()
    except sqlite3.Error:
        return default
    return row[0] if row and row[0] is not None else default


def _tree_bytes(root: str) -> int:
    total = 0
    for d, _ds, fs in os.walk(root):
        for f in fs:
            try:
                total += os.path.getsize(os.path.join(d, f))
            except OSError:
                pass
    return total


def index_health(db: str, measure_bytes: bool = True,
                 scan_vectors: bool = True) -> dict:
    """Is this index of a package actually usable? Read-only.

    `ok` is decided from the database alone (cheap enough per request):
    chunks > 0, defs >= MIN_DEFS, and at least one chunk from a code file
    rather than documentation. The byte figures need the fetched source on
    disk and are for reporting; `measure_bytes=False` skips them, and
    `scan_vectors=False` skips reading every vector (62 MB on three.js).

    `zero_vectors` counts chunks whose vector is all zeros -- an index built
    with INDEX_NO_EMBED=1 is all zeros by design and still serves the symbol
    table and BM25, so it does not affect `ok`; `vectors` says which it is.
    """
    out = {"db": db, "exists": os.path.exists(db), "chunks": 0, "files": 0,
           "code_chunks": 0, "defs": 0, "zero_vectors": 0, "vectors": "none",
           "complete": False, "bytes_indexed": None, "bytes_fetched": None,
           "ratio": None, "ok": False}
    if not out["exists"]:
        return out
    try:
        con = sqlite3.connect(f"file:{os.path.abspath(db)}?mode=ro", uri=True)
    except sqlite3.Error:
        return out
    try:
        out["chunks"] = _scalar(con, "SELECT COUNT(*) FROM chunks")
        out["files"] = _scalar(con, "SELECT COUNT(DISTINCT path) FROM chunks")
        doc = " AND ".join(f"lower(path) NOT LIKE '%{e}'" for e in sorted(_DOC_EXTS))
        out["code_chunks"] = _scalar(con, f"SELECT COUNT(*) FROM chunks WHERE {doc}")
        out["defs"] = _scalar(con, "SELECT COUNT(*) FROM defs")
        dim = _scalar(con, "SELECT length(vec) FROM chunks LIMIT 1")
        if dim and scan_vectors:
            out["zero_vectors"] = _scalar(
                con, f"SELECT COUNT(*) FROM chunks WHERE vec = zeroblob({int(dim)})")
        if dim and scan_vectors:
            # Unit norm on a sample of the non-zero vectors: "it returned
            # 200" is not aliveness, "it returned a unit-norm vector" is.
            import numpy as np
            rows = con.execute(
                f"SELECT vec FROM chunks WHERE vec != zeroblob({int(dim)}) "
                "ORDER BY random() LIMIT 200").fetchall()
            if rows:
                norms = [float(np.linalg.norm(np.frombuffer(r[0], dtype=np.float32)))
                         for r in rows]
                out["norm_min"] = round(min(norms), 4)
                out["norm_max"] = round(max(norms), 4)
                out["norm_ok"] = all(abs(x - 1.0) < 0.01 for x in norms)
        n, z = out["chunks"], out["zero_vectors"]
        out["vectors"] = ("unscanned" if not scan_vectors else "none"
                          if n == 0 or z == n else "all" if z == 0 else "partial")
        try:
            meta = dict(con.execute("SELECT k, v FROM meta").fetchall())
        except sqlite3.Error:
            meta = {}
        out["complete"] = meta.get("complete") == "1"
        out["selector"] = meta.get("selector")
        try:
            roots = [r[0] for r in con.execute("SELECT path FROM roots")]
        except sqlite3.Error:
            roots = []
        paths = ([r[0] for r in con.execute("SELECT DISTINCT path FROM chunks")]
                 if measure_bytes else [])
    finally:
        con.close()

    out["ok"] = (out["chunks"] > 0 and out["defs"] >= MIN_DEFS
                 and out["code_chunks"] > 0)
    if measure_bytes:
        root = next((r for r in roots if os.path.isdir(r)), None)
        if root:
            got = 0
            for rel in paths:
                try:
                    got += os.path.getsize(os.path.join(root, rel))
                except OSError:
                    pass
            out["bytes_indexed"] = got
            out["bytes_fetched"] = _tree_bytes(root)
        else:
            for k in ("bytes_indexed", "bytes_fetched"):
                if meta.get(k, "").isdigit():
                    out[k] = int(meta[k])
        if out["bytes_indexed"] is not None and out["bytes_fetched"]:
            out["ratio"] = round(out["bytes_indexed"] / out["bytes_fetched"], 4)
    return out


def needs_index(name: str, version: str) -> bool:
    """True when there is no usable index for this exact version."""
    return not index_health(db_path(name, version), measure_bytes=False,
                            scan_vectors=False)["ok"]


# ------------------------------------------------------------- indexing
SELECTOR = "code_files/1"
# The same ceiling index_code.check_alive applies (INDEX_MAX_DEAD, 0.02).
MAX_ZERO_FRACTION = float(os.environ.get("INDEX_MAX_DEAD", "0.02"))
STACK = os.environ.get("LLAMA_STACK_URL", "http://127.0.0.1:11434")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "embeddings")


def probe_embeddings() -> str | None:
    """None if the embedding server returns a real vector, else why not.

    PROTOCOL rule 1: "it returned 200" is not aliveness; a non-zero vector is.
    Asked before a run so a dead server fails in one second, not after
    index_code has spent its retries.
    """
    body = json.dumps({"model": EMBED_MODEL, "input": ["function probe() {}"]})
    req = urllib.request.Request(f"{STACK}/v1/embeddings", data=body.encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            vec = json.load(r)["data"][0]["embedding"]
    except Exception as e:                                       # noqa: BLE001
        return f"{STACK}/v1/embeddings ({EMBED_MODEL!r}) failed: {type(e).__name__}: {e}"
    if not vec or not any(abs(x) > 0 for x in vec):
        return f"{STACK}/v1/embeddings returned a zero or empty vector"
    return None


def _write_meta(db: str, rows: dict, files: list[str], src: str) -> None:
    con = sqlite3.connect(db)
    try:
        con.execute("CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT)")
        con.executemany(
            "INSERT INTO meta(k, v) VALUES(?, ?) "
            "ON CONFLICT(k) DO UPDATE SET v = excluded.v",
            [(k, str(v)) for k, v in rows.items()])
        # The selection, so a symbol-only rebuild (scripts/reindex_symbols.py)
        # re-reads the same files instead of re-walking with repo rules.
        con.execute("CREATE TABLE IF NOT EXISTS package_files(path TEXT PRIMARY KEY)")
        con.execute("DELETE FROM package_files")
        con.executemany("INSERT OR IGNORE INTO package_files VALUES(?)",
                        [(os.path.relpath(f, src).replace("\\", "/"),) for f in files])
        con.commit()
    finally:
        con.close()


def index_package(name: str, version: str, embed: bool = False,
                  db: str | None = None, src: str | None = None,
                  timeout: int | None = None, log=None) -> dict:
    """Fetch, select with code_files, index, and install only if healthy.

    Built into `<db>.building` and moved into place at the end, so nothing
    reads a half-written index and a failed run leaves the old one intact.
    An index that fails index_health is NOT installed: that is the exact
    README-only shape this exists to stop being counted as held.
    """
    import subprocess
    import sys

    say = log or (lambda *_a: None)
    src = src or fetch(name, version)
    if not src:
        return {"ok": False, "installed": False,
                "error": f"could not fetch {name}@{version} from {REGISTRY}"}
    if embed:
        why = probe_embeddings()
        if why:
            return {"ok": False, "installed": False, "error": why}

    files = code_files(src)
    target = db or db_path(name, version)
    os.makedirs(os.path.dirname(os.path.abspath(target)), exist_ok=True)
    tmp = target + ".building"
    listing = tmp + ".files"
    for p in (tmp, tmp + "-journal"):
        if os.path.exists(p):
            os.remove(p)
    with open(listing, "w", encoding="utf-8") as f:
        f.write("\n".join(files))
    say(f"{name}@{version}: {len(files)} files selected from {src}")

    env = dict(os.environ)
    env["CODE_INDEX_DB"] = tmp
    env["INDEX_FILES"] = listing
    env.pop("INDEX_APPEND", None)
    if embed:
        env.pop("INDEX_NO_EMBED", None)
        env["LLAMA_STACK_URL"] = STACK
    else:
        env["INDEX_NO_EMBED"] = "1"
    script = os.path.join(HERE, "..", "scripts", "index_code.py")
    try:
        proc = subprocess.run([sys.executable, script, src], env=env,
                              capture_output=True, text=True,
                              encoding="utf-8", errors="replace",
                              timeout=timeout or (14400 if embed else 1800))
    except subprocess.TimeoutExpired:
        return {"ok": False, "installed": False, "error": "index_code timed out"}
    finally:
        try:
            os.remove(listing)
        except OSError:
            pass
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-6:]
        return {"ok": False, "installed": False,
                "error": f"index_code exited {proc.returncode}: " + " | ".join(tail)}

    selected = sum(os.path.getsize(f) for f in files if os.path.exists(f))
    _write_meta(tmp, {"selector": SELECTOR, "package": f"{name}@{version}",
                      "mode": "src" if has_src(src) else "built",
                      "files_selected": len(files), "bytes_selected": selected,
                      "bytes_fetched": _tree_bytes(src),
                      "embedded": "1" if embed else "0"}, files, src)
    health = index_health(tmp)
    health["db"] = target
    health["files_selected"] = len(files)
    health["bytes_selected"] = selected
    why = None
    if not health["ok"]:
        why = (f"chunks={health['chunks']} defs={health['defs']} "
               f"code_chunks={health['code_chunks']}")
    elif embed and (health["zero_vectors"] > MAX_ZERO_FRACTION * health["chunks"]
                    or not health.get("norm_ok")):
        # PROTOCOL rule 1: an --embed run is only installed if its vectors
        # are real -- non-zero, and unit norm on a sample.
        why = (f"embeddings failed: {health['zero_vectors']}/{health['chunks']} "
               f"zero, sampled norms {health.get('norm_min')}..{health.get('norm_max')}")
    if why:
        os.remove(tmp)
        health["installed"] = False
        health["error"] = "refused to install: " + why
        return health
    try:
        os.replace(tmp, target)
    except OSError as e:
        health["installed"] = False
        health["error"] = (f"built at {tmp} but could not replace {target} "
                           f"({e}); something has it open")
        return health
    health["installed"] = True
    return health


def _split_spec(spec: str) -> tuple[str, str]:
    """`@scope/pkg@1.2.3` -> ('@scope/pkg', '1.2.3')."""
    name, at, version = spec.rpartition("@")
    if not at or not name or not version:
        raise SystemExit(f"expected <name>@<version>, got {spec!r}")
    return name, version


def _main(argv: list[str]) -> int:
    if argv and argv[0] == "index":
        import argparse
        ap = argparse.ArgumentParser(prog="deps.py index")
        ap.add_argument("spec", nargs="+", help="name@version")
        ap.add_argument("--embed", action="store_true",
                        help=f"embed via {STACK} (else INDEX_NO_EMBED=1)")
        a = ap.parse_args(argv[1:])
        bad = 0
        for spec in a.spec:
            name, version = _split_spec(spec)
            h = index_package(name, version, embed=a.embed, log=print)
            print(json.dumps(h, indent=2))
            bad += not h.get("installed")
        return 1 if bad else 0
    if argv and argv[0] == "health":
        bad = 0
        for spec in argv[1:]:
            db = spec if spec.endswith(".sqlite3") else db_path(*_split_spec(spec))
            h = index_health(db)
            print(json.dumps(h, indent=2))
            bad += not h["ok"]
        return 1 if bad else 0
    r = argv[0] if argv else os.getcwd()
    for row in plan(r):
        state = f"{row['chunks']} chunks" if row["chunks"] else "not indexed"
        print(f"  {row['name']:<28} {row['version']:<12} "
              f"{row['imports']:>4} imports   {state}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_main(sys.argv[1:]))
