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


if __name__ == "__main__":
    import sys
    r = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
    for row in plan(r):
        state = f"{row['chunks']} chunks" if row["chunks"] else "not indexed"
        print(f"  {row['name']:<28} {row['version']:<12} "
              f"{row['imports']:>4} imports   {state}")
