#!/usr/bin/env python
"""Rebuild defs/refs only, leaving chunks and their vectors untouched.

WHY THIS EXISTS SEPARATELY FROM index_code.py

The two halves of an index are built by completely different machinery. Chunks
and their embeddings need the GPU and take twenty minutes for three.js; defs
and refs are tree-sitter and sqlite, need no network at all, and take seconds.

When the symbol EXTRACTOR changes -- as it just did, to capture exported
bindings -- only the second half is stale. Re-running the full indexer would
re-embed 15,021 unchanged chunks to fix a table that does not depend on them,
and would compete for the GPU with whatever experiment is running.

WHAT PROMPTED IT

`export const positionLocal = ...` was not being recorded as a definition.
three.js builds nearly the whole TSL surface that way, so the most-used API in
the corpus had zero definitions and hundreds of references each. Fixing the
extractor is worthless until the indexes are rebuilt with it, and rebuilding
them cannot be allowed to cost a GPU-hour.

SAFETY

defs and refs are deleted and rewritten inside one transaction. chunks, vec and
roots are never touched, and the completion marker is preserved: this run does
not re-establish that the chunk side is complete, so it must not claim to.
"""
from __future__ import annotations

import os
import sqlite3
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import symbols as sym  # noqa: E402
from index_code import iter_files  # noqa: E402


def source_root(con: sqlite3.Connection) -> str | None:
    try:
        row = con.execute("SELECT path FROM roots LIMIT 1").fetchone()
    except sqlite3.Error:
        return None
    return row[0] if row and row[0] and os.path.isdir(row[0]) else None


def rebuild(db: str) -> dict:
    con = sqlite3.connect(db)
    try:
        root = source_root(con)
        if not root:
            return {"db": db, "skipped": "source tree not on disk"}
        before = con.execute("SELECT COUNT(*) FROM defs").fetchone()[0]
        t0 = time.time()

        d_rows: list[tuple] = []
        r_rows: list[tuple] = []
        n_files = 0
        # A package index built by deps.index_package records the files it
        # selected (its code may be only in dist/, which the repo walk
        # prunes). Re-walking it with repo rules would strip every
        # definition that lives in a bundle.
        try:
            listed = [os.path.join(root, r[0]) for r in
                      con.execute("SELECT path FROM package_files")]
        except sqlite3.Error:
            listed = None
        for path in (listed if listed else iter_files(root)):
            rel = os.path.relpath(path, root).replace("\\", "/")
            d, r = sym.extract(path, rel)
            d_rows += d
            r_rows += r
            n_files += 1

        # One transaction: a crash here must not leave an index with no
        # symbols at all, which would look exactly like a repository that
        # defines nothing.
        con.execute("BEGIN")
        con.execute("DELETE FROM defs")
        con.execute("DELETE FROM refs")
        con.executemany("INSERT INTO defs VALUES(?,?,?,?,?,?)", d_rows)
        con.executemany("INSERT INTO refs VALUES(?,?,?,?,?)", r_rows)
        con.commit()
        return {"db": os.path.basename(db), "files": n_files,
                "defs_before": before, "defs_after": len(d_rows),
                "refs": len(r_rows), "secs": round(time.time() - t0, 1)}
    finally:
        con.close()


def main() -> None:
    targets = sys.argv[1:]
    if not targets:
        root = os.path.join(HERE, "..", "index")
        for sub in ("repos", "packages"):
            d = os.path.join(root, sub)
            if os.path.isdir(d):
                targets += [os.path.join(d, f) for f in sorted(os.listdir(d))
                            if f.endswith(".sqlite3")]
    if not targets:
        raise SystemExit("no indexes found")

    print(f"  rebuilding symbols in {len(targets)} indexes "
          f"(tree-sitter only, no GPU)\n")
    for db in targets:
        try:
            r = rebuild(db)
        except Exception as e:                                   # noqa: BLE001
            print(f"  {os.path.basename(db):<28} FAILED {type(e).__name__}: {e}")
            continue
        if r.get("skipped"):
            print(f"  {r['db'] if 'db' in r else os.path.basename(db):<28} "
                  f"skipped: {r['skipped']}")
            continue
        delta = r["defs_after"] - r["defs_before"]
        print(f"  {r['db']:<28} {r['files']:>5} files  "
              f"defs {r['defs_before']:>6} -> {r['defs_after']:<6} "
              f"({delta:+d})  refs {r['refs']:<7} {r['secs']}s")


if __name__ == "__main__":
    main()
