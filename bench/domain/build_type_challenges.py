#!/usr/bin/env python
"""Import the type-challenges corpus as a CONTAMINATED TypeScript subset.

Source: https://github.com/type-challenges/type-challenges (MIT, see
tasks_type_challenges/LICENSE.type-challenges and NOTICE). TypeHero material
(AGPL-3.0) is NOT used anywhere here.

The upstream repo is cloned into _work/tc-src (gitignored). For every
challenge in questions/ this writes

  tasks_type_challenges/<id>/template.ts     upstream, verbatim
  tasks_type_challenges/<id>/test-cases.ts   upstream, verbatim
  tasks_type_challenges/<id>/reference.ts    ours -- upstream ships none

and one row per INCLUDED challenge to tasks_type_challenges.jsonl. A
challenge is included only if it has a reference.ts and is not listed in
tasks_type_challenges/excluded.json ({id: reason}). test_grade_tc.py is the
proof that every included reference passes and the template does not.

Contamination: type-challenges has been public since 2020 and its community
solutions are all over GitHub; every row is `contaminated: true`.

    git clone --depth 1 https://github.com/type-challenges/type-challenges \\
        bench/domain/_work/tc-src
    python bench/domain/build_type_challenges.py
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import grade  # noqa: E402

SRC = os.environ.get("TC_SRC", os.path.join(grade.WORK, "tc-src"))
OUT_DIR = grade.TC_DIR
OUT_JSONL = grade.TC_JSONL
EXCLUDED = os.path.join(OUT_DIR, "excluded.json")
REPO = "https://github.com/type-challenges/type-challenges"
_DIR = re.compile(r"^(\d+)-(warm|easy|medium|hard|extreme)-(.+)$")


def read(p: str) -> str:
    with open(p, encoding="utf-8") as f:
        return f.read()


def write(p: str, s: str) -> None:
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(s)


def description(readme: str) -> str:
    """The English README between the generated header and footer.

    The header is title/badges/author/links; the footer is Back / Share /
    "Check out Solutions" / related-challenge links -- dropped so the prompt
    carries no pointer to solutions."""
    a = readme.find("<!--info-header-end-->")
    b = readme.find("<!--info-footer-start-->")
    if a < 0 or b < 0 or b < a:
        raise ValueError("README without the generated header/footer markers")
    body = readme[a + len("<!--info-header-end-->"):b].strip()
    # "[476 - Sum](https://tsch.js.org/476)" names a sibling challenge, but the
    # site it links hosts every solution. Keep the name, drop the link.
    return re.sub(r"\[([^\]]+)\]\(https?://tsch\.js\.org[^)]*\)", r"\1", body)


def identifiers(code: str) -> set[str]:
    root = grade.parse(code, "typescript")
    out, stack = set(), [root]
    while stack:
        n = stack.pop()
        if n.type in ("identifier", "type_identifier"):
            out.add(grade._txt(n))
        stack.extend(n.children)
    return out


def prompt(num: int, title: str, diff: str, desc: str, template: str) -> str:
    return (
        f"TypeScript type challenge #{num}: {title} ({diff}).\n\n"
        f"{desc}\n\n"
        "Start from this template and replace the placeholder implementation. "
        "Keep every name it declares; add or constrain type parameters where "
        "the task needs them.\n\n"
        f"```ts\n{template.rstrip()}\n```\n\n"
        "Your code is type-checked with `tsc --strict` as its own file, next "
        "to test cases that use these names directly and compare types with "
        "`Equal`/`Expect`-style helpers from `@type-challenges/utils`. Answer "
        "with a single ```ts code block containing the complete "
        "implementation, including any helper types it needs.")


def collect(src: str = SRC, write_files: bool = True) -> dict:
    """Every upstream challenge -> {rows, missing, stale, by, sha}.

    rows holds the INCLUDED challenges (reference.ts present, not excluded)."""
    qdir = os.path.join(src, "questions")
    sha = subprocess.run(["git", "-C", src, "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    if write_files:
        os.makedirs(OUT_DIR, exist_ok=True)
        shutil.copy2(os.path.join(src, "LICENSE"),
                     os.path.join(OUT_DIR, "LICENSE.type-challenges"))
        os.makedirs(os.path.join(OUT_DIR, "utils"), exist_ok=True)
        for fn in ("index.d.ts", "package.json"):
            shutil.copy2(os.path.join(src, "utils", fn),
                         os.path.join(OUT_DIR, "utils", fn))
    excluded = json.loads(read(EXCLUDED)) if os.path.isfile(EXCLUDED) else {}

    rows, missing, seen = [], [], set()
    by: dict = {}
    for d in sorted(os.listdir(qdir)):
        m = _DIR.match(d)
        if not m:
            continue
        num, diff, slug = int(m.group(1)), m.group(2), m.group(3)
        tid = f"tc-{m.group(1)}-{slug}"
        seen.add(tid)
        qsrc = os.path.join(qdir, d)
        info = yaml.safe_load(read(os.path.join(qsrc, "info.yml"))) or {}
        template = read(os.path.join(qsrc, "template.ts"))
        tests = read(os.path.join(qsrc, "test-cases.ts"))
        dst = os.path.join(OUT_DIR, tid)
        if write_files:
            os.makedirs(dst, exist_ok=True)
            write(os.path.join(dst, "template.ts"), template)
            write(os.path.join(dst, "test-cases.ts"), tests)
        declared = grade.tc_names(template)["declared"]
        used = identifiers(tests)
        entry = sorted(declared & used) or sorted(declared)
        state = ("excluded" if tid in excluded else
                 "included" if os.path.isfile(os.path.join(dst, "reference.ts"))
                 else "missing")
        by.setdefault(diff, {"included": 0, "excluded": 0, "missing": 0})
        by[diff][state] += 1
        if state == "missing":
            missing.append(tid)
        if state != "included":
            continue
        rel = lambda fn: f"tasks_type_challenges/{tid}/{fn}"   # noqa: E731
        rows.append({
            "id": tid,
            "domain": "typescript_tc",
            "prompt": prompt(num, info.get("title", slug), diff,
                             description(read(os.path.join(qsrc, "README.md"))),
                             template),
            "grader": {"kind": "tc_tests", "test": rel("test-cases.ts"),
                       "template": rel("template.ts"), "entry": entry},
            "reference": rel("reference.ts"),
            "contaminated": True,
            "needs_retrieval": False,
            "difficulty": diff,
            "upstream": {"repo": REPO, "commit": sha, "dir": f"questions/{d}",
                         "title": info.get("title"),
                         "tags": info.get("tags"),
                         "info_difficulty": info.get("difficulty")},
        })
    return {"rows": rows, "missing": missing, "by": by, "sha": sha,
            "seen": seen, "stale": sorted(set(excluded) - seen)}


def main() -> int:
    if not os.path.isdir(os.path.join(SRC, "questions")):
        print(f"no upstream checkout at {SRC}; clone {REPO} there first")
        return 2
    c = collect()
    with open(OUT_JSONL, "w", encoding="utf-8", newline="\n") as f:
        for r in c["rows"]:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"upstream {c['sha']}: {len(c['seen'])} challenges, "
          f"{len(c['rows'])} rows -> {OUT_JSONL}")
    for k in ("warm", "easy", "medium", "hard", "extreme"):
        if k in c["by"]:
            print(f"  {k:8s} {c['by'][k]}")
    if c["missing"]:
        print(f"  {len(c['missing'])} with no reference and no exclusion reason:")
        for t in c["missing"]:
            print("    ", t)
    if c["stale"]:
        print("  excluded.json names unknown ids:", c["stale"])
    return 1 if c["missing"] or c["stale"] else 0


if __name__ == "__main__":
    sys.exit(main())
