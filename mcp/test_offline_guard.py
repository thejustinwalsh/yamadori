#!/usr/bin/env python
"""The offline guard, asserted. No GPU, and no connection to the stack.

scripts/run_tests.py runs every offline suite with scripts/offline_guard/
first on PYTHONPATH and YAMADORI_OFFLINE_GUARD=1: its sitecustomize refuses a
connect to any port the stack serves on and logs it, and run_tests fails the
suite. Found 2026-09-27: twelve offline suites read llama-swap's /props or
/running, or the embedder, and llama-swap's /upstream/bonsai/props LOADS
bonsai -- it cut an engine test window short twice.

Each check runs a child Python with the guard's environment. A refused
connect raises before any packet leaves (the audit hook runs before the
syscall), so these checks never touch the live stack. Also checked: the
offending suites pin the served state (mcp/served_fixture.py) and the pinned
fixture parses.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
GUARD = os.path.join(ROOT, "scripts", "offline_guard")
sys.path.insert(0, HERE)

import served_fixture  # noqa: E402

_results: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> bool:
    _results.append((bool(ok), name, detail))
    return bool(ok)


def child(code: str, guard: bool = True, state: str | None = None
          ) -> tuple[int, str, list[dict]]:
    """Run `code` in a child Python with the guard's environment; `state`
    replaces the live-state roots (a temp tree, so no check here can touch
    the real index/)."""
    fd, log = tempfile.mkstemp(prefix="guard_test_", suffix=".jsonl")
    os.close(fd)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(
        p for p in (GUARD, os.environ.get("PYTHONPATH", "")) if p),
        YAMADORI_OFFLINE_GUARD_LOG=log)
    env["YAMADORI_OFFLINE_GUARD"] = "1" if guard else "0"
    if state:
        env["YAMADORI_OFFLINE_GUARD_STATE"] = state
    try:
        r = subprocess.run([sys.executable, "-X", "utf8", "-c", code],
                           capture_output=True, text=True, env=env, timeout=60)
        with open(log, encoding="utf-8") as f:
            rows = [json.loads(ln) for ln in f if ln.strip()]
    finally:
        os.remove(log)
    return r.returncode, (r.stdout or "") + (r.stderr or ""), rows


def test_a_stack_port_is_refused_and_logged():
    for port in (11434, 1234, 1235, 1237, 8888, 10001):
        rc, out, rows = child(
            "import urllib.request\n"
            "try:\n"
            f"    urllib.request.urlopen('http://127.0.0.1:{port}/props', timeout=2)\n"
            "    print('CONNECTED')\n"
            "except Exception as e:\n"
            "    print('refused:', type(getattr(e, 'reason', e)).__name__)\n")
        check("refused: OfflineGuardRefused" in out and "CONNECTED" not in out
              and len(rows) == 1 and rows[0]["port"] == port,
              f"a connect to :{port} is refused before it leaves, and logged",
              out.strip()[-300:] + " " + json.dumps(rows)[:200])


def test_a_raw_socket_and_asyncio_are_refused():
    rc, out, rows = child(
        "import socket, asyncio\n"
        "s = socket.socket()\n"
        "try:\n"
        "    s.connect_ex(('localhost', 11434)); print('CONNECTED')\n"
        "except ConnectionRefusedError as e: print('raw refused')\n"
        "async def m():\n"
        "    try:\n"
        "        await asyncio.open_connection('127.0.0.1', 1234); print('CONNECTED')\n"
        "    except ConnectionRefusedError: print('async refused')\n"
        "asyncio.run(m())\n")
    check("raw refused" in out and "async refused" in out
          and "CONNECTED" not in out and len(rows) == 2,
          "connect_ex on a raw socket and asyncio's connect are refused too",
          out.strip()[-300:])


def test_a_fake_on_an_ephemeral_port_is_allowed():
    rc, out, rows = child(
        "import threading, urllib.request\n"
        "from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer\n"
        "class H(BaseHTTPRequestHandler):\n"
        "    def do_GET(self):\n"
        "        self.send_response(200); self.end_headers(); self.wfile.write(b'ok')\n"
        "    def log_message(self, *a): pass\n"
        "srv = ThreadingHTTPServer(('127.0.0.1', 0), H)\n"
        "threading.Thread(target=srv.serve_forever, daemon=True).start()\n"
        "print(urllib.request.urlopen(f'http://127.0.0.1:{srv.server_port}/', "
        "timeout=5).read().decode())\n")
    check(out.strip().endswith("ok") and not rows,
          "a suite's own fake on port 0 still answers", out.strip()[-200:])


def test_the_guard_is_inert_unless_switched_on():
    rc, out, rows = child("import sys; print('asyncio' in sys.modules)",
                          guard=False)
    check(out.strip().endswith("False") and not rows,
          "with YAMADORI_OFFLINE_GUARD off it installs nothing (asyncio is "
          "not even imported)", out.strip())


def test_run_tests_turns_it_on_offline_only():
    src = open(os.path.join(ROOT, "scripts", "run_tests.py"),
               encoding="utf-8").read()
    check('offline_env["YAMADORI_OFFLINE_GUARD"] = "1"' in src
          and "guard_hits(" in src and "GUARD_DIR" in src,
          "run_tests.py sets the guard for offline suites and fails a suite "
          "whose log has a row")


def _live_tree() -> str:
    """A temp stand-in for index/: a database with a row, a json file."""
    import sqlite3
    d = tempfile.mkdtemp(prefix="guard_live_")
    con = sqlite3.connect(os.path.join(d, "store.sqlite3"))
    con.execute("CREATE TABLE t(x)")
    con.execute("INSERT INTO t VALUES(1)")
    con.commit()
    con.close()
    with open(os.path.join(d, "state.json"), "w", encoding="utf-8") as f:
        f.write("{}")
    return d


def test_a_write_to_the_live_state_is_refused_and_logged():
    live = _live_tree()
    js = os.path.join(live, "state.json").replace("\\", "/")
    rc, out, rows = child(
        "import os, json, pathlib\n"
        "for name, f in (\n"
        f"    ('open', lambda: open({js!r}, 'w')),\n"
        f"    ('append', lambda: open({js!r}, 'a')),\n"
        f"    ('pathlib', lambda: pathlib.Path({js!r}).write_text('x')),\n"
        f"    ('new file', lambda: open({live!r} + '/new.txt', 'x')),\n"
        f"    ('replace', lambda: os.replace({js!r}, {js!r} + '.bak')),\n"
        f"    ('remove', lambda: os.remove({js!r})),\n"
        f"    ('makedirs', lambda: os.makedirs({live!r}, exist_ok=True)),\n"
        f"    ('read', lambda: open({js!r}).read())):\n"
        "    try:\n"
        "        r = f(); print(name, 'ALLOWED', repr(r)[:20])\n"
        "    except PermissionError as e: print(name, 'refused')\n",
        state=live)
    got = dict(ln.split(" ", 1) for ln in out.splitlines()
               if ln.split(" ", 1)[0] in ("open", "append", "pathlib", "new",
                                          "replace", "remove", "read",
                                          "makedirs"))
    check(all(got.get(k) == "refused" for k in
              ("open", "append", "pathlib", "replace", "remove"))
          and "file refused" in out and got.get("read", "").startswith(
              "ALLOWED") and got.get("makedirs", "").startswith("ALLOWED"),
          "a writing open, a rename and a remove under a live root are "
          "refused; a read, and makedirs(exist_ok) of a directory that is "
          "there, are not", out.strip()[-500:])
    check(len(rows) == 6 and all(r["kind"] == "write" for r in rows)
          and all(r.get("path") and r.get("op") for r in rows),
          "each refusal is logged as a write, with its path and operation",
          json.dumps(rows)[:300])
    with open(os.path.join(live, "state.json"), encoding="utf-8") as f:
        check(f.read() == "{}", "and the file is untouched")


def test_sqlite_reads_pass_and_writes_are_refused():
    live = _live_tree()
    db = os.path.join(live, "store.sqlite3").replace("\\", "/")
    rc, out, rows = child(
        "import sqlite3\n"
        f"c = sqlite3.connect({db!r})\n"
        "print('read', c.execute('SELECT COUNT(*) FROM t').fetchone()[0])\n"
        "for sql in ('INSERT INTO t VALUES(2)', 'DELETE FROM t',\n"
        "            'CREATE TABLE IF NOT EXISTS t(x)', 'ALTER TABLE t ADD y',\n"
        "            'CREATE TABLE IF NOT EXISTS u(x)', 'CREATE INDEX i ON t(x)',\n"
        "            'PRAGMA journal_mode=wal', 'PRAGMA user_version=3',\n"
        "            'CREATE TEMP TABLE s(x)', 'INSERT INTO s VALUES(1)',\n"
        "            'PRAGMA busy_timeout=100', 'PRAGMA table_info(t)'):\n"
        "    try:\n"
        "        c.execute(sql); print('ok', sql)\n"
        "    except sqlite3.DatabaseError as e: print('refused', sql)\n"
        "c.commit(); c.close()\n"
        "from sqlite3 import connect\n"
        f"d = connect({db!r})\n"
        "try:\n"
        "    d.execute('UPDATE t SET x=9'); print('ok update')\n"
        "except sqlite3.DatabaseError: print('refused update')\n"
        f"r = sqlite3.connect('file:' + {db!r} + '?mode=ro', uri=True)\n"
        "print('ro', r.execute('SELECT COUNT(*) FROM t').fetchone()[0])\n"
        "import tempfile, os\n"
        "f = sqlite3.connect(os.path.join(tempfile.mkdtemp(), 'x.db'))\n"
        "f.execute('CREATE TABLE a(x)'); f.execute('INSERT INTO a VALUES(1)')\n"
        "print('free ok')\n", state=live)
    refused = [ln[8:] for ln in out.splitlines() if ln.startswith("refused ")]
    allowed = [ln[3:] for ln in out.splitlines() if ln.startswith("ok ")]
    check("read 1" in out and "ro 1" in out and "free ok" in out,
          "a read-write connection may read, a read-only one too, and a "
          "database outside the live roots is untouched by the guard",
          out.strip()[-400:])
    check(sorted(refused) == sorted([
        "INSERT INTO t VALUES(2)", "DELETE FROM t",
        "CREATE TABLE IF NOT EXISTS u(x)", "CREATE INDEX i ON t(x)",
        "ALTER TABLE t ADD y",
        "PRAGMA journal_mode=wal", "PRAGMA user_version=3", "update"])
          and {"CREATE TEMP TABLE s(x)", "INSERT INTO s VALUES(1)",
               "PRAGMA busy_timeout=100", "PRAGMA table_info(t)",
               "CREATE TABLE IF NOT EXISTS t(x)"} <= set(allowed),
          "every statement that would change the file is refused (a "
          "`from sqlite3 import connect` too); temp tables, read pragmas "
          "and CREATE ... IF NOT EXISTS of a table that is there (a no-op: "
          "code_search opens every package index so) pass",
          out.strip()[-600:])
    check(len(rows) >= 7 and all(r["kind"] == "write"
                                 and r["op"].startswith("sqlite ")
                                 for r in rows),
          "and each is logged", json.dumps(rows)[:300])
    import sqlite3
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        state = (con.execute("SELECT x FROM t").fetchall(),
                 con.execute("PRAGMA journal_mode").fetchone()[0],
                 con.execute("PRAGMA user_version").fetchone()[0])
    finally:
        con.close()
    check(state == ([(1,)], "delete", 0),
          "the database is exactly as it was", state)


def _run_tests_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "run_tests_under_test", os.path.join(ROOT, "scripts", "run_tests.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_snapshot_layer_attributes_changes():
    import sqlite3
    import time
    rt = _run_tests_module()
    root = tempfile.mkdtemp(prefix="guard_root_")
    os.makedirs(os.path.join(root, "index", "skills", "lib"))
    os.makedirs(os.path.join(root, "logs"))
    files = {"index/jobs.sqlite3": b"x", "index/pairs.jsonl": b"{}\n",
             "index/skills/lib/SKILL.md": b"# s\n",
             "logs/deploy_check.jsonl": b"{}\n",
             "index/concept_seed_last.json": b"{}"}
    for rp, data in files.items():
        with open(os.path.join(root, rp), "wb") as f:
            f.write(data)
    rt.ROOT = root
    st = rt.LiveState()
    check(st.diff() == [], "nothing changed, nothing reported")
    t0 = time.time()
    for rp in ("index/pairs.jsonl", "logs/deploy_check.jsonl",
               "index/concept_seed_last.json", "index/skills/lib/SKILL.md"):
        with open(os.path.join(root, rp), "ab") as f:
            f.write(b"more\n")
    os.utime(os.path.join(root, "index", "jobs.sqlite3"), None)   # touch
    changed = st.diff()
    paths = sorted(p for p, _w in changed)
    check(paths == sorted(["index/pairs.jsonl", "logs/deploy_check.jsonl",
                           "index/concept_seed_last.json",
                           "index/skills/lib/SKILL.md"]),
          "a content change is found by sha; a touch with the same bytes "
          "is not a change", changed)
    fails, notes = rt.attribute(changed, t0, time.time(), set())
    check(len(fails) == 3 and len(notes) == 1
          and "concept_seed_last" in notes[0]
          and any("SKILL.md" in f and "no" in f for f in fails),
          "a file the stack writes is a note; a static file, the deploy "
          "verdicts, and a skill file with no worker job active FAIL",
          (fails, notes))
    # A worker job active in the window: the skill file is not charged.
    os.remove(os.path.join(root, "index", "jobs.sqlite3"))
    con = sqlite3.connect(os.path.join(root, "index", "jobs.sqlite3"))
    con.execute("CREATE TABLE jobs(id TEXT, queue TEXT, started REAL, "
                "finished REAL, heartbeat REAL)")
    con.execute("INSERT INTO jobs VALUES('j1','skill.arm',?,?,NULL)",
                (t0 - 1, time.time() + 1))
    con.commit()
    con.close()
    fails, notes = rt.attribute([("index/skills/lib/SKILL.md", "x")], t0,
                                time.time(), set())
    check(not fails and notes and "skill.arm" in notes[0],
          "the same change while a worker job ran is a note", notes)
    fails, _n = rt.attribute([("logs/deploy_check.jsonl", "x")], t0,
                             time.time(), {"logs/deploy_check.jsonl"})
    check(not fails, "a path the write guard already named is not "
                     "reported twice")
    # index/max_mode.json is the proxy's max-mode lease state
    # (mcp/max_mode.py _write_state), rewritten while the stack serves: 16
    # suites were failed for it on 2026-09-29 with no suite writing it.
    fails, notes = rt.attribute([("index/max_mode.json", "x")], t0,
                                time.time(), set())
    check(not fails and len(notes) == 1 and "max-mode" in notes[0],
          "index/max_mode.json (the running proxy's max_mode state) is a "
          "note, never a failure", (fails, notes))
    src = open(os.path.join(ROOT, "scripts", "run_tests.py"),
               encoding="utf-8").read()
    check("LiveState()" in src and "attribute(" in src
          and "changed the live state" in src,
          "run_tests.py runs the snapshot around every offline suite")


def test_the_pinned_served_state():
    p = served_fixture.props()
    import tiers
    check(isinstance(p.get("n_ctx"), int) and p["n_ctx"] > 0
          and isinstance(p.get("total_slots"), int) and p["total_slots"] > 0
          and "<|im_start|>" in p["chat_template"],
          "the pinned /props parses: n_ctx, total_slots and the template",
          json.dumps({k: v for k, v in p.items() if k != "chat_template"})[:300])
    check(tiers.efforts_in_template(p["chat_template"])
          == ("xhigh", "medium", "low"),
          "and the template's guard clause names the efforts",
          str(tiers.efforts_in_template(p["chat_template"])))
    cfg = open(os.path.join(ROOT, "config.yaml"), encoding="utf-8").read()
    import re
    check(re.search(rf"^\s*-c {p['n_ctx']}\s*$", cfg, re.M) is not None,
          "n_ctx is the -c config.yaml launches bonsai with",
          f"n_ctx={p['n_ctx']}")


def main() -> int:
    for fn in (test_a_stack_port_is_refused_and_logged,
               test_a_raw_socket_and_asyncio_are_refused,
               test_a_fake_on_an_ephemeral_port_is_allowed,
               test_the_guard_is_inert_unless_switched_on,
               test_run_tests_turns_it_on_offline_only,
               test_a_write_to_the_live_state_is_refused_and_logged,
               test_sqlite_reads_pass_and_writes_are_refused,
               test_the_snapshot_layer_attributes_changes,
               test_the_pinned_served_state):
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
    print(f"\n{'=' * 70}\n  {passed}/{len(_results)} checks passed")
    return 0 if passed == len(_results) else 1


if __name__ == "__main__":
    sys.exit(main())
