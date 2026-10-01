#!/usr/bin/env python
"""The project-path rule, the command readers and the compaction counters
(mcp/progress.py), asserted directly. No GPU, no network.

    python mcp/test_progress.py      -> "N/M checks passed"

WHAT THIS IS GATING (docs/SELF-IMPROVEMENT-LOG.md #52, #53, #56):

  1. PROJECT PATHS. /tmp, a dot-file, a scratch-harness name
     (progress.SCRATCH_NAMES) and a write outside a known working directory
     are not the project's; a file the task names always is.
  2. COMMANDS. inspect_only: a command that only looks; the read tools'
     table (tool_code.READ_KNOWN).
  3. COMPACTIONS. A compaction makes the work log due once.
  4. REMOVED 2026-09-29 with deep thinking (docs/REMOVED.md): `learn` (the
     project learned from a step's writes, the inferred working directory)
     and its tests; the step kind went 2026-09-27.
  5. REMOVED 2026-09-27 (operator: task-targeted steering in prompts;
     skills are the channel): the progress line (#50), the unchanged-read
     line (#54) and the work log's files-read listing are gone, and so is
     the state they kept.

The end-to-end path (the ledger, the served template, x_yamadori) is gated in
mcp/test_ledger.py.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

_TMP = tempfile.mkdtemp(prefix="yamadori_test_progress_")
os.environ["YAMADORI_NEBARI_DB"] = os.path.join(_TMP, "nebari.sqlite3")
os.environ["YAMADORI_CORPUS_DB"] = os.path.join(_TMP, "corpus.sqlite3")

import progress  # noqa: E402
import tool_code  # noqa: E402

_results: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> bool:
    _results.append((bool(ok), name, detail))
    return bool(ok)


def call(name: str, args: dict, cid: str) -> dict:
    return {"id": cid, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


READ = '{"content": "1|const PLAYER = {};", "total_lines": 1}'
OK_WRITE = '{"bytes_written": 42}'


def test_project_paths():
    proj = {"root": "/workspace/app", "named": ["js/player.js", "index.html"]}
    cases = [("js/player.js", True), ("/workspace/app/js/game.js", True),
             ("/tmp/harness.js", False), ("/tmp/pw/test-game.js", False),
             (".runtime_test.js", False), ("js/.dbg1.js", False),
             ("test_full.js", False), ("diag2.js", False),
             ("/root/other/x.js", False), ("css/styles.css", True),
             ("C:\\Users\\me\\AppData\\Local\\Temp\\x.js", False)]
    got = {p: progress.is_project(p, proj) for p, _ in cases}
    check(all(got[p][0] is want for p, want in cases),
          "project paths: /tmp, dot-files, scratch-harness names and writes "
          "outside the known working directory are not the project's",
          json.dumps({p: v for p, v in got.items()}))
    named = {"root": None, "named": ["test_full.js"]}
    check(progress.is_project("test_full.js", named)[0],
          "a file the task names is the project's, whatever its name")
    msgs = [{"role": "user", "content":
             "Fix `PLAYER.hit is not a function` at js/enemies.js:212 and the "
             "score in js/game.js; see /etc/hosts."}]
    check(progress.named_paths(msgs) == ["js/enemies.js", "js/game.js"],
          "the task's named files: relative paths with an extension, "
          "file:line read as the file, absolute paths not",
          json.dumps(progress.named_paths(msgs)))
    sysm = [{"role": "system", "content": "Your working directory is "
             "/workspace/octo. Be brief."}]
    check(progress.stated_root(sysm) == "/workspace/octo",
          "a stated working directory is the project root",
          str(progress.stated_root(sysm)))


def test_compaction_counters():
    check(not progress.reinject_due("acct", "L5"),
          "no compaction yet: nothing due")
    progress.note_compaction("acct", "L5")
    due = progress.reinject_due("acct", "L5")
    progress.mark_reinjected("acct", "L5")
    check(due and not progress.reinject_due("acct", "L5"),
          "a compaction makes the work log due once; marked, it is not due "
          "again until the next compaction")


def test_inspect_only():
    """progress.step_kind (#53's by-result thinking cap) was removed
    2026-09-27 (docs/CONSTANTS-AUDIT.md); inspect_only stays, because
    skill_select reads it."""
    check(progress.inspect_only("cd /w && ls -la js && sed -n 1,20p js/a.js")
          and not progress.inspect_only("node test.js")
          and not progress.inspect_only("cat a.js > b.js")
          and not hasattr(progress, "step_kind"),
          "inspect_only: a look-only command; a run or a redirect is not; "
          "step_kind is gone")


def test_read_table():
    got = {n: tool_code.read_target(call(n, a, "x")) for n, a in (
        ("read_file", {"path": "./js/a.js", "offset": 5, "limit": 10}),
        ("read", {"filePath": "src/x.ts"}),
        ("Read", {"file_path": "/w/a.py"}),
        ("write_file", {"path": "a.js", "content": "x"}),
        ("terminal", {"command": "cat a.js"}))}
    check(got["read_file"] == {"path": "js/a.js", "range": {"limit": 10,
                                                            "offset": 5},
                               "tool": "read_file"}
          and got["read"]["path"] == "src/x.ts"
          and got["Read"]["path"] == "/w/a.py"
          and got["write_file"] is None and got["terminal"] is None,
          "tool_code.READ_KNOWN: Hermes, OpenCode, Pi and Claude Code read "
          "tools, their path and range; a write or a shell cat is not a read "
          "tool", json.dumps(got))


TREE_TASK = ("Build a tile game.\n\nPROJECT STRUCTURE:\n"
             "tile-game/\n"
             "  index.html        -- canvas and script tags\n"
             "  css/main.css      -- fullscreen canvas\n"
             "  js/\n"
             "    board.js        -- the grid\n"
             "    main.js         -- the loop\n\n"
             "In main.js, call BOARD.reset() on start.")


def test_tree_paths():
    got = progress.named_paths([{"role": "user", "content": TREE_TASK}])
    check(got == ["index.html", "css/main.css", "js/board.js", "js/main.js"],
          "a project tree's files come qualified by their folders, relative "
          "to its one top folder; the bare main.js in the prose is the tree's "
          "js/main.js, not a second name", json.dumps(got))
    glyphs = ("Layout:\n\n"
              "app/\n"
              "├── index.html\n"
              "├── src/\n"
              "│   ├── game.ts   # the loop\n"
              "│   └── util/\n"
              "│       └── rng.ts\n"
              "└── package.json\n")
    got = progress.tree_paths(glyphs)[0]
    check(got == ["index.html", "src/game.ts", "src/util/rng.ts",
                  "package.json"],
          "a tree drawn with box glyphs, nested folders", json.dumps(got))
    flat = ("Files:\n\nindex.html\njs/\n  a.js\n  b.js\ncss/\n  s.css\n")
    got = progress.tree_paths(flat)[0]
    check(got == ["index.html", "js/a.js", "js/b.js", "css/s.css"],
          "a tree with several top entries is relative to the tree itself "
          "(no project folder to strip)", json.dumps(got))
    prose = ("Fix the score.\n- in game.js the total is wrong\n"
             "- see js/ui.js line 40\n")
    got = progress.tree_paths(prose)[0]
    check(got == [], "bullet prose is not a tree", json.dumps(got))


def main() -> int:
    for fn in (test_project_paths, test_compaction_counters,
               test_inspect_only, test_read_table, test_tree_paths):
        print(f"\n--- {fn.__name__} ---")
        n0 = len(_results)
        try:
            fn()
        except Exception:                                        # noqa: BLE001
            check(False, f"{fn.__name__} itself raised",
                  traceback.format_exc().strip().split("\n")[-1])
            traceback.print_exc()
        for ok, name, detail in _results[n0:]:
            print(("  pass  " if ok else "  FAIL  ") + name
                  + (f"   <- {detail}" if not ok and detail else ""))
    passed = sum(1 for ok, _, _ in _results if ok)
    total = len(_results)
    print(f"\n{'=' * 70}\n  {passed}/{total} checks passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
