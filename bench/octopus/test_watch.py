#!/usr/bin/env python
"""The Octopus watcher's repair count, its imitated-note detector and its
survival of a dead parent. No GPU, no network, no live run.

    python bench/octopus/test_watch.py      -> "N/M checks passed"

v0b-V0-xhigh-1 (2026-09-25): the watcher said "repairs: 1" while the
transcript showed several "Repaired ..." notes. The relay's x_yamadori.
tool_code records ONE repair at that point (step 8, enemies.js, 4 errors,
fix-up written); the other "Repaired" lines were written by the MODEL,
imitating the notes in its history (steps 9, 15, 19, 23: tool_code clean,
0 errors, no fix-up). So: repairs are counted per file from the records
(errors_before > 0 and a fix-up written), and an unbacked "Repaired" line
is its own defect. The watcher also died with the agent that started it;
it now survives a closed stdout and has --detach.
"""
from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import watch  # noqa: E402

_results: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> bool:
    _results.append((bool(ok), name, detail))
    return bool(ok)


RUN = "t-watch-1"
P = "/root/space-shooter/js/"


def _tc(path: str, before: int, written: int, kind: str = "file") -> dict:
    return {"enabled": True, "fix": True, "errors_before": before, "errors_after": 0,
            "rounds": 1 if written else 0, "stopped": "fixed" if written else "clean",
            "fixup": {"units": 1, "written": written} if written else None,
            "files": [{"path": path, "kind": kind, "errors_before": before,
                       "errors_after": 0}]}


def _row(t0: float, tc: dict) -> dict:
    return {"method": "POST", "path": "/v1/chat/completions", "status": 200,
            "t0": t0, "t_end": t0 + 10,
            "request": {"n_messages": 3, "last_role": "tool"},
            "response": {"finish_reason": "tool_calls",
                         "usage": {"prompt_tokens": 1000, "completion_tokens": 100},
                         "tool_calls": ["write_file"], "content_chars": 10,
                         "reasoning_chars": 10,
                         "x_yamadori": {"tool_code": tc, "cache": {"reused": 990}}}}


def _text(ts: float, text: str) -> dict:
    return {"type": "text", "text": text, "timestamp": int(ts * 1000)}


def fixture(root: str) -> None:
    logs = os.path.join(root, "logs", RUN)
    os.makedirs(logs)
    os.makedirs(os.path.join(root, "runs", RUN))
    rows = [_row(1000.0, _tc(P + "enemies.js", 4, 1)),      # a real repair
            _row(2000.0, _tc(P + "player.js", 0, 0)),       # clean
            _row(3000.0, _tc(P + "enemies.js", 0, 0, "edit")),
            _row(4000.0, _tc(P + "game.js", 2, 1))]         # a second real one
    with open(os.path.join(logs, "relay.jsonl"), "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    ev = [_text(1005, f"\n\nRepaired {P}enemies.js (javascript): 4 syntax errors fixed "
                      "in 1 round; prettier would change 69 lines; sent as written."),
          # the MODEL's own content, imitating the note (v0b step 9)
          _text(2005, f"Enemy factory done.\n\nRepaired {P}player.js (javascript): 4 "
                      "syntax errors fixed in 1 round; sent as written.\n\n"),
          _text(2006, f"\n\nVerified {P}player.js (javascript): parses; sent as "
                      "written."),
          _text(3005, f"Repaired {P}enemies.js (javascript): 1 syntax error fixed in "
                      "2 rounds; sent as written."),
          _text(4005, f"\n\nRepaired {P}game.js (javascript): 2 syntax errors fixed in "
                      "1 round (the repaired file was sent; formatting left as "
                      "written).")]
    with open(os.path.join(logs, "hermes.jsonl"), "w", encoding="utf-8") as f:
        f.writelines(json.dumps(e) + "\n" for e in ev)
    with open(os.path.join(logs, "proxy_offset"), "w") as f:
        f.write("0")


def test_repairs_are_counted_per_file_from_the_records():
    root = tempfile.mkdtemp(prefix="octo_watch_")
    fixture(root)
    watch.runmod.LOGS_DIR = os.path.join(root, "logs")
    watch.runmod.RUNS_DIR = os.path.join(root, "runs")
    watch.PROXY_LOG = os.path.join(root, "proxy.out.log")
    open(watch.PROXY_LOG, "w").close()
    watch.STATE_DB = os.path.join(root, "no-state.db")
    a = watch.assess(RUN)
    s = a["summary"]
    check(s.get("repairs_by_file") == {P + "enemies.js": 1, P + "game.js": 1}
          and s["repairs"] == 2,
          "every tool_code record with errors_before > 0 and a fix-up written, "
          "counted per file", json.dumps({k: s.get(k) for k in ("repairs",
                                                                 "repairs_by_file")}))
    imit = [d for d in a["defects"] if d["kind"] == "imitated_note"]
    check([d["step"] for d in imit] == [2, 3]
          and "player.js" in imit[0]["evidence"] and "enemies.js" in imit[1]["evidence"],
          "a 'Repaired' line no record of its step backs is an imitated_note "
          "(the model's words); the backed ones and 'Verified' are not",
          json.dumps(imit)[:400])


def test_it_survives_its_parent():
    real = sys.stdout

    class Closed(io.StringIO):
        def write(self, *_a):
            raise OSError(22, "Invalid argument")      # a pipe whose reader died
    sys.stdout = Closed()
    try:
        watch.say("a defect line")
        survived = True
    except Exception:                                            # noqa: BLE001
        survived = False
    finally:
        sys.stdout = real
    check(survived, "printing to a closed stdout does not kill the watcher")
    seen: dict = {}

    class FakePopen:
        def __init__(self, cmd, **kw):
            seen.update(cmd=cmd, **kw)
            self.pid = 4242
    import subprocess
    orig = subprocess.Popen
    subprocess.Popen = FakePopen
    try:
        d = tempfile.mkdtemp(prefix="octo_detach_")
        pid = watch.detach([RUN, "--wait", "--detach"], d)
    finally:
        subprocess.Popen = orig
    flags = seen.get("creationflags", 0)
    check(pid == 4242 and "--detach" not in seen["cmd"] and RUN in seen["cmd"]
          and open(os.path.join(d, "watch.pid")).read() == "4242"
          and seen.get("stdin") == subprocess.DEVNULL
          and (flags & 0x01000008 == 0x01000008 if os.name == "nt"
               else seen.get("start_new_session")),
          "--detach re-launches itself outside the process tree and job "
          "(DETACHED_PROCESS | BREAKAWAY_FROM_JOB / setsid), output to "
          "watch.out, pid to watch.pid", json.dumps({k: str(v) for k, v in seen.items()}))


def main() -> int:
    for fn in (test_repairs_are_counted_per_file_from_the_records,
               test_it_survives_its_parent):
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
    print(f"\n{'=' * 70}\n  {passed}/{len(_results)} checks passed")
    return 0 if passed == len(_results) else 1


if __name__ == "__main__":
    sys.exit(main())
