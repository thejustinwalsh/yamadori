#!/usr/bin/env python
"""The project-path rule, the project learned from a step's writes, the step
kind and the compaction counters (mcp/progress.py), asserted directly. No
GPU, no network.

    python mcp/test_progress.py      -> "N/M checks passed"

WHAT THIS IS GATING (docs/SELF-IMPROVEMENT-LOG.md #52, #53, #56):

  1. PROJECT PATHS. /tmp, a dot-file, a scratch-harness name
     (progress.SCRATCH_NAMES) and a write outside a known working directory
     are not the project's; a file the task names always is.
  2. THE PROJECT LEARNED. `learn` reads a step's successful writes into the
     project (the working directory inferred from a named file, widened,
     never over a stated one); a failed write is not a write; a retry
     changes nothing; it writes no text.
  3. STEP KIND. read/search/inspect -> read; a failing result -> error.
  4. COMPACTIONS. A compaction makes the work log due once.
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


class Convo:
    """A Hermes-shaped conversation, one step at a time."""

    def __init__(self, task: str, lineage: str):
        self.msgs = [{"role": "system", "content": "You are Hermes Agent."},
                     {"role": "user", "content": task}]
        self.lineage = lineage
        self.n = 0

    def step(self, calls: list[tuple[str, dict, str]], t: float = 0.0
             ) -> dict:
        cs = []
        for name, args, _res in calls:
            self.n += 1
            cs.append(call(name, args, f"c{self.n}"))
        self.msgs.append({"role": "assistant", "content": "",
                          "tool_calls": cs})
        for c, (_n, _a, res) in zip(cs, calls):
            self.msgs.append({"role": "tool", "tool_call_id": c["id"],
                              "content": res})
        return progress.learn("acct", self.lineage, self.msgs)


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


def test_learn_writes_and_idempotence():
    c = Convo("Build the game: index.html, js/game.js, js/player.js.", "L1")
    r = c.step([("write_file", {"path": "/workspace/app/js/player.js",
                                "content": "const PLAYER = {};"}, OK_WRITE),
                ("write_file", {"path": "/tmp/harness.js", "content": "x"},
                 OK_WRITE),
                ("write_file", {"path": "test_full.js", "content": "x"},
                 OK_WRITE)])
    check(r["project_writes"] == ["/workspace/app/js/player.js"]
          and [w["path"] for w in r["scratch_writes"]]
          == ["/tmp/harness.js", "test_full.js"]
          and r["root"] == "/workspace/app" and r["root_by"] == "inferred",
          "a step's writes: the project's and the scratch ones, and the "
          "working directory inferred from the first write of a named file",
          json.dumps(r))
    st = progress.load("acct", "L1")
    again = progress.learn("acct", "L1", c.msgs)
    check(again == r and progress.load("acct", "L1") == st,
          "a retry of the same request changes nothing",
          json.dumps([again, st]))
    check(set(r) <= {"project_writes", "scratch_writes", "root", "root_by",
                     "named"},
          "learn returns a record, never text for the model", json.dumps(r))


def test_failed_write_is_not_a_write():
    c = Convo("Fix js/player.js.", "L2")
    r = c.step([("patch", {"path": "/w/js/player.js", "old_string": "zz",
                           "new_string": "yy"},
                 '{"success": false, "error": "Could not find a match for '
                 'old_string"}')])
    check(not r["project_writes"] and not r["root"],
          "a patch that failed is not a write, and teaches no root",
          json.dumps(r))


def test_the_situations_are_gone():
    gone = ("PROGRESS_LINE", "PROGRESS_LINE_NONE", "UNCHANGED_LINE",
            "PROGRESS_STEPS", "SITUATION_HEAD", "WORK_LOG_READS_HEAD",
            "observe", "situation_text", "work_log_reads", "terminal_write")
    check(not [n for n in gone if hasattr(progress, n)],
          "no situation line, switch or generator is left in progress.py",
          json.dumps([n for n in gone if hasattr(progress, n)]))
    import tiers
    check(not {"progress_note", "unchanged_read"} & set(tiers.BEHAVIOURS)
          and all("situations" not in t for t in tiers.TIERS.values()),
          "the switches and the `situations` tier flag are gone")
    # A conversation whose state was written by the old observe: its
    # retired fields are dropped on the next save; the project stays.
    progress.save("acct", "OLD", {"named": ["js/a.js"], "root": "/w",
                                  "root_by": "inferred", "step": 9,
                                  "since": 8, "since_t": 1.0,
                                  "reads": {"x": {}}, "seen": [{"k": "k"}],
                                  "last_write": {"path": "js/a.js"},
                                  "compactions": 1})
    c = Convo("Fix js/a.js.", "OLD")
    r = c.step([("write_file", {"path": "/w/js/a.js", "content": "x"},
                 OK_WRITE)])
    st = progress.load("acct", "OLD")
    check(r["root"] == "/w" and st.get("compactions") == 1
          and not {"step", "since", "since_t", "reads", "seen",
                   "last_write"} & set(st),
          "an old state: the retired fields go, the project and the "
          "compaction count stay", json.dumps(st))


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


def test_root_from_tree_and_widening():
    # The v0f-V0 shape: the tree names js/board.js, the model writes
    # /workspace/tile-game/js/board.js -> the root is the project folder.
    c = Convo(TREE_TASK, "R1")
    t = 6_000_000.0
    c.step([("write_file", {"path": "/workspace/tile-game/js/board.js",
                            "content": "const BOARD = {};\n"}, OK_WRITE)], t)
    st = progress.load("acct", "R1")
    check(st.get("root") == "/workspace/tile-game",
          "the working directory from a tree-qualified name is the project "
          "folder, not the written file's own folder (v0f-V0: "
          "/workspace/space-shooter/js)", json.dumps(st.get("root")))
    r = c.step([("write_file", {"path": "/workspace/tile-game/serve.js",
                                "content": "x"}, OK_WRITE)], t + 60)
    check(r["project_writes"] == ["/workspace/tile-game/serve.js"],
          "a file the task does not name, outside js/ but inside the project "
          "folder, is a project write", json.dumps(r))
    # Bare names only: the first write sets js/, the next named file one
    # level up widens it to the common folder.
    c2 = Convo("Write board.js and index.html for a tile game.", "R2")
    c2.step([("write_file", {"path": "/w/tile/js/board.js", "content": "x"},
              OK_WRITE)], t)
    first = progress.load("acct", "R2").get("root")
    r = c2.step([("write_file", {"path": "/w/tile/index.html",
                                 "content": "<script src='js/board.js'>"
                                            "</script>"}, OK_WRITE)], t + 60)
    second = progress.load("acct", "R2").get("root")
    check(first == "/w/tile/js" and second == "/w/tile"
          and r["project_writes"] == ["/w/tile/index.html"],
          "a root inferred from a bare name is widened to the deepest common "
          "folder when the next named file lands outside it",
          json.dumps([first, second, r]))
    r = c2.step([("write_file", {"path": "/w/tile/tools/build.js",
                                 "content": "x"}, OK_WRITE)], t + 120)
    check(r["project_writes"] == ["/w/tile/tools/build.js"],
          "after widening, an unnamed file in the project folder counts",
          json.dumps(r))
    c3 = Convo("Write board.js and index.html.", "R3")
    c3.step([("write_file", {"path": "/a/js/board.js", "content": "x"},
              OK_WRITE)], t)
    c3.step([("write_file", {"path": "/b/index.html", "content": "x"},
              OK_WRITE)], t + 60)
    check(progress.load("acct", "R3").get("root") == "/a/js",
          "never widened to the filesystem root",
          json.dumps(progress.load("acct", "R3").get("root")))
    c4 = Convo("Write board.js and index.html.", "R4")
    c4.msgs[0]["content"] += " Your working directory is /w/tile/js."
    r = c4.step([("write_file", {"path": "/w/tile/index.html",
                                 "content": "x"}, OK_WRITE),
                 ("write_file", {"path": "/w/tile/other.js",
                                 "content": "x"}, OK_WRITE)], t)
    check(r["root"] == "/w/tile/js"
          and r["root_by"] == "stated"
          and r["project_writes"] == ["/w/tile/index.html"],
          "a stated working directory is never widened (a named file outside "
          "it still counts; an unnamed one does not)", json.dumps(r))


def test_html_page_write():
    c = Convo("Build index.html and js/game.js.", "H1")
    t = 7_000_000.0
    c.step([("write_file", {"path": "/w/app/js/game.js", "content": "x"},
             OK_WRITE)], t)
    r = c.step([("write_file", {"path": "/w/app/index.html", "content":
                                "<!DOCTYPE html>\n<html><body><canvas>"
                                "</canvas>\n<script src=\"js/game.js\">"
                                "</script></body></html>\n"}, OK_WRITE)],
               t + 100)
    check(r["project_writes"] == ["/w/app/index.html"],
          "an .html page with only src scripts is a project write "
          "(tool_code.detect whole_pages; v0f-V0's index.html was not "
          "progress)", json.dumps(r))
    d = tool_code.detect(call("write_file", {"path": "a.html", "content":
                                             "<script src='x.js'></script>"},
                              "h"))
    check(d["units"] == [],
          "the code check still sees no code in that page (whole_pages is "
          "only for which file a call writes)", json.dumps(d["units"]))


def _responses(items: list[dict]) -> list[dict]:
    import responses_api
    chat, _ctx = responses_api.to_chat({
        "model": "yamadori", "instructions": "You are Hermes Agent.",
        "input": items, "store": False, "stream": True,
        "prompt_cache_key": "pck_test"})
    return chat["messages"]


def test_responses_path():
    """The Responses wire (Hermes codex_responses, the v0f-V0 run): items
    through responses_api.to_chat, then progress -- the call's name and the
    result's text must pair by call_id."""
    board = '{"content": "1|const BOARD = {};\\n2|export default BOARD;"}'
    items = [{"role": "user", "content": "Fix js/board.js."}]
    n = [0]

    def step(calls):
        ids = []
        for name, args, _out in calls:
            n[0] += 1
            ids.append(f"call_0f3a9c21b7e4_{n[0]:08x}")
            items.append({"type": "function_call", "call_id": ids[-1],
                          "name": name, "arguments": json.dumps(args)})
        for cid, (_n, _a, out) in zip(ids, calls):
            items.append({"type": "function_call_output", "call_id": cid,
                          "output": out})
        msgs = _responses(items)
        return msgs, progress.learn("acct", "RS", msgs)
    msgs, r1 = step([("read_file", {"path": "/w/app/js/board.js"}, board)])
    tools = [m for m in msgs if m.get("role") == "tool"]
    asst = [m for m in msgs if m.get("role") == "assistant"]
    check(tools and asst and tools[-1]["tool_call_id"]
          == asst[-1]["tool_calls"][0]["id"]
          and asst[-1]["tool_calls"][0]["function"]["name"] == "read_file"
          and progress._text(tools[-1]) == board,
          "to_chat: the tool message's tool_call_id is the assistant call's "
          "id, the call keeps its name, the result its text",
          json.dumps(msgs[-2:])[:400])
    items.append({"type": "message", "role": "assistant", "status":
                  "completed", "content": [{"type": "output_text",
                                            "text": "Patching."}]})
    _m, r4 = step([("patch", {"path": "/w/app/js/board.js",
                              "old_string": "{}", "new_string": "{ n: 1 }"},
                    '{"success": true, "diff": "..."}')])
    check(r4["project_writes"] == ["/w/app/js/board.js"]
          and not r1["project_writes"],
          "Responses wire: the patch pairs with its result by call_id and is "
          "a project write; the read is not", json.dumps([r1, r4]))


def main() -> int:
    for fn in (test_project_paths, test_learn_writes_and_idempotence,
               test_failed_write_is_not_a_write, test_the_situations_are_gone,
               test_compaction_counters, test_inspect_only, test_read_table, test_tree_paths,
               test_root_from_tree_and_widening, test_html_page_write,
               test_responses_path):
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
