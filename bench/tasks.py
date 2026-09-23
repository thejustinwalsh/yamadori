#!/usr/bin/env python
"""Generate a large benchmark set with ground truth taken from the code itself.

WHY GENERATED AND NOT HAND-WRITTEN

A hand-written benchmark is small, and a small benchmark cannot separate a real
effect from noise. Twenty tasks cannot distinguish a 60% system from a 75% one
at any confidence worth reporting, and the honest move when you only have
twenty is to say you do not know yet.

Ground truth for code questions does not have to be written by hand, because
the code already contains it. `defs` says exactly where a symbol is defined;
`refs` says exactly which files use it. Those are facts about the repository,
extracted mechanically, so the label cannot drift from the code and cannot be
shaded by whoever is hoping for a result.

That is what makes the set large: 12,244 definitions in three.js alone, 578 in
koota. The constraint is not how many tasks exist, it is how many are FAIR.

WHAT MAKES A TASK FAIR

Most symbols make bad questions, and including them measures the wrong thing:

  ambiguous     a name defined in six places has no single right answer, so
                scoring it by exact path punishes a correct response
  trivial       one or two characters, or a common English word, matches
                everything and tests tokenisation rather than retrieval
  vendored      minified bundles under examples/jsm/libs are not the library,
                and a benchmark that rewards finding chevrotain.module.min.js
                is training the system to give worse answers
  unreachable   defined in a file the indexer skipped, so no system could
                answer it and the task only adds noise

Each filter below throws away tasks that would flatter or unfairly punish the
system rather than measure it. The rejection counts are reported, because a
generator that silently discards 90% of its candidates is selecting for
something and the reader should be able to see what.

POSED AS A REMOTE CALLER WOULD POSE THEM

Each task carries a short code snippet with real imports, because that is what
actually arrives over the wire. It also means the benchmark exercises library
discovery rather than assuming the server was told where to look.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sqlite3
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "mcp"))

# Words that are also identifiers. A question about `map` or `length` measures
# nothing about retrieval.
COMMON = {
    "map", "set", "get", "add", "remove", "value", "index", "length", "name",
    "type", "data", "size", "count", "list", "item", "next", "prev", "start",
    "end", "min", "max", "sum", "key", "node", "text", "line", "path", "file",
    "time", "color", "scale", "clone", "copy", "apply", "call", "test", "main",
    "init", "update", "render", "draw", "clear", "reset", "state", "props",
}

# Not the library. A definition here is a third-party bundle that happens to
# live in the tree.
VENDORED = re.compile(
    r"(?:^|/)(?:node_modules|vendor|third_party|dist|build)/"
    r"|\.min\.(?:js|mjs|cjs)$"
    r"|examples/jsm/libs/", re.I)

TEST_FILE = re.compile(r"(?:^|/)(?:tests?|__tests__|spec)/|\.(?:test|spec)\.[jt]sx?$", re.I)


def _rows(db: str, sql: str, args: tuple = ()) -> list:
    con = sqlite3.connect(db)
    try:
        return con.execute(sql, args).fetchall()
    finally:
        con.close()


def _plausible(name: str) -> bool:
    if len(name) < 4 or len(name) > 40:
        return False
    if name.lower() in COMMON:
        return False
    if not re.match(r"^[A-Za-z_$][\w$]*$", name):
        return False
    # A name that is only digits-and-underscores, or screaming constants, tends
    # to be generated code rather than an API anyone asks about.
    return not name.isupper() or len(name) > 8


def locate_tasks(db: str, source: str, snippet: str, want: int,
                 rnd: random.Random) -> tuple[list[dict], dict]:
    """"Where is X defined?" -- ground truth is the one path that defines it."""
    rej: dict[str, int] = {}

    def drop(why: str) -> None:
        rej[why] = rej.get(why, 0) + 1

    rows = _rows(db, "SELECT name, kind, path, line, COUNT(*) OVER "
                     "(PARTITION BY name) AS n FROM defs")
    pool = []
    for name, kind, path, line, n in rows:
        if not _plausible(name):
            drop("implausible name")
        elif n > 1:
            drop("ambiguous, defined more than once")
        elif VENDORED.search(path):
            drop("vendored or minified")
        elif TEST_FILE.search(path):
            drop("defined in a test")
        else:
            pool.append({"name": name, "kind": kind, "path": path, "line": line})

    rnd.shuffle(pool)
    tasks = []
    for p in pool[:want]:
        tasks.append({
            "id": f"locate:{source}:{p['name']}",
            "kind": "locate",
            "source": source,
            "symbol": p["name"],
            "prompt": (f"{snippet}\n\nWhere is `{p['name']}` defined? "
                       f"Answer with the file path and line number."),
            "truth": {"path": p["path"], "line": p["line"], "decl": p["kind"]},
        })
    return tasks, rej


def reference_tasks(db: str, source: str, snippet: str, want: int,
                    rnd: random.Random) -> tuple[list[dict], dict]:
    """"Which files use X?" -- ground truth is the set of referencing paths.

    Bounded at both ends on purpose. A symbol used in two files is a question
    about one lookup; one used in four hundred is a question about whether the
    system will dump four hundred lines, and neither measures retrieval.
    """
    rej: dict[str, int] = {}

    def drop(why: str) -> None:
        rej[why] = rej.get(why, 0) + 1

    rows = _rows(db, "SELECT name, COUNT(DISTINCT path) FROM refs "
                     "GROUP BY name HAVING COUNT(DISTINCT path) BETWEEN 3 AND 25")
    pool = []
    for name, _n in rows:
        if not _plausible(name):
            drop("implausible name")
            continue
        paths = sorted({
            p for (p,) in _rows(db, "SELECT DISTINCT path FROM refs WHERE name=?",
                                (name,))
            if not VENDORED.search(p)})
        if len(paths) < 3:
            drop("all uses were vendored")
            continue
        pool.append({"name": name, "paths": paths})

    rnd.shuffle(pool)
    tasks = []
    for p in pool[:want]:
        tasks.append({
            "id": f"refs:{source}:{p['name']}",
            "kind": "references",
            "source": source,
            "symbol": p["name"],
            "prompt": (f"{snippet}\n\nWhich files reference `{p['name']}`? "
                       f"List the file paths."),
            "truth": {"paths": p["paths"]},
        })
    return tasks, rej


# The imports a caller would actually have open when asking about each source.
# Real specifiers, because discovery is part of what is under test.
SNIPPETS = {
    "three": ("```ts\nimport * as THREE from 'three/webgpu'\n"
              "import { Fn, vec3, positionLocal, uniform } from 'three/tsl'\n```"),
    "typegpu": ("```ts\nimport tgpu from 'typegpu'\n"
                "import { struct, vec3f, f32 } from 'typegpu/data'\n```"),
    "koota": ("```ts\nimport { createWorld, trait } from 'koota'\n"
              "import { useQuery } from 'koota/react'\n```"),
    "glyph": "```ts\nimport { Glyph } from '@pmndrs/glyph'\n```",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-source", type=int, default=60,
                    help="locate tasks per source; reference tasks are half this")
    ap.add_argument("--seed", type=int, default=20260921)
    ap.add_argument("--out", default=os.path.join(HERE, "tasks.jsonl"))
    args = ap.parse_args()

    import repos

    sources: list[tuple[str, str]] = []
    for e in repos.known():
        sources.append((os.path.basename(e["root"]).lower(), repos.db_path(e["root"])))
    pkg_dir = os.path.join(HERE, "..", "index", "packages")
    if os.path.isdir(pkg_dir):
        for fn in sorted(os.listdir(pkg_dir)):
            if fn.endswith(".sqlite3") and "@" in fn:
                name = fn[:-len(".sqlite3")].rpartition("@")[0]
                sources.append((name.lower(), os.path.join(pkg_dir, fn)))

    rnd = random.Random(args.seed)
    all_tasks: list[dict] = []
    print(f"{'source':<12} {'locate':>7} {'refs':>7}   top rejections")
    print("-" * 72)
    for source, db in sources:
        if not os.path.exists(db):
            continue
        snippet = SNIPPETS.get(source, f"```ts\nimport '{source}'\n```")
        loc, rej_l = locate_tasks(db, source, snippet, args.per_source, rnd)
        ref, rej_r = reference_tasks(db, source, snippet, args.per_source // 2, rnd)
        all_tasks += loc + ref
        merged: dict[str, int] = {}
        for d in (rej_l, rej_r):
            for k, v in d.items():
                merged[k] = merged.get(k, 0) + v
        top = ", ".join(f"{k} {v}" for k, v in
                        sorted(merged.items(), key=lambda x: -x[1])[:3])
        print(f"{source:<12} {len(loc):>7} {len(ref):>7}   {top}")

    rnd.shuffle(all_tasks)
    with open(args.out, "w", encoding="utf-8") as f:
        for t in all_tasks:
            f.write(json.dumps(t) + "\n")
    by_kind: dict[str, int] = {}
    for t in all_tasks:
        by_kind[t["kind"]] = by_kind.get(t["kind"], 0) + 1
    print("-" * 72)
    print(f"  {len(all_tasks)} tasks  {by_kind}")
    print(f"  written {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
