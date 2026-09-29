"""THE OFFLINE GUARD: an offline test suite never reaches the live stack.

scripts/run_tests.py puts this directory first on PYTHONPATH for every
OFFLINE suite (never a live one) and sets YAMADORI_OFFLINE_GUARD=1, so Python
imports this file at startup -- in the suite and in every Python process it
starts. It refuses any TCP connect to a port the stack serves on, whatever
the host (the proxy listens on 0.0.0.0, so a LAN address reaches it too):

    1234 proxy, 1235 tools API, 1237/1238 Laya, 8888 SearXNG,
    11434 llama-swap, 10001-10099 llama-swap's model servers (startPort)

Why (2026-09-27): offline suites read the served chat template through
llama-swap's /upstream/bonsai/props (tiers.accepted_efforts and friends),
which LOADS `bonsai` when it is unloaded -- it cut an engine test window
short twice (docs/ENGINES.md, "The PTQ1_0 PDL race"). A suite that silently
falls back when the stack is down also passes or fails depending on whether
it is up: not a test (PROTOCOL rule 3).

A refused connect raises ConnectionRefusedError (the code under test sees an
outage, as it would with the stack down) AND is appended to the file
YAMADORI_OFFLINE_GUARD_LOG names, with the frames that asked. run_tests.py
fails the suite when that file is non-empty, so a fallback cannot hide the
attempt. The hook is an audit hook (sys.addaudithook, `socket.connect`): it
sees every socket.connect / connect_ex, however the caller reached them, and
cannot be removed by the code it watches. asyncio's Windows proactor connects
through _overlapped.ConnectEx, which raises no audit event, so its connect is
wrapped too.

Nothing happens unless YAMADORI_OFFLINE_GUARD=1: a PYTHONPATH left pointing
here by accident changes nothing.

AND NEVER WRITES A LIVE STORE (2026-09-27). The same audit hook refuses --
and logs, with the frames that asked -- every WRITE to the live state: any
path under index/ (the databases, the skill library, the package registry
history, the trigger cache, the router labels, the media store ...) and
under logs/ (deploy_check.jsonl's verdicts). Found 2026-09-27: 21 rows of
index/jobs.sqlite3's skill_fallbacks (2026-09-26 14:28 to 09-27 01:06) were
an offline suite's fixture request (mcp/test_domains.py through
proxy.prepare, with no YAMADORI_JOBS_DB), and the worker then LEARNED from
them. What counts as a write:

  - `open` with a writing mode or flags (w, a, x, +; O_WRONLY, O_RDWR,
    O_CREAT, O_TRUNC, O_APPEND) -- builtins.open, io.open, os.open, pathlib,
    numpy.savez, shutil.copy all raise it;
  - os.rename / os.replace (either end), os.remove / unlink, os.rmdir,
    os.mkdir, os.truncate, os.utime, os.chmod, shutil.rmtree;
  - a WRITE STATEMENT on a read-write sqlite3 connection to a live
    database. A read-only URI (`file:...?mode=ro`, `immutable=1`) needs
    nothing -- the way a suite should read a live database (mcp/
    test_utility.py's corpus replay does). A read-write one (the package
    readers in code_search / deps open that way) may still READ: the guard
    wraps sqlite3.connect and installs, on each such connection, an
    authorizer that refuses, at prepare time,
    and logs every statement that would change the file -- INSERT, UPDATE,
    DELETE, CREATE / DROP / ALTER (so a module's CREATE TABLE IF NOT EXISTS
    on open counts: it is a writer's open), ANALYZE, REINDEX, ATTACH, and
    the PRAGMAs that change the file (journal_mode=, user_version=, ...);
    temp-schema writes are allowed -- and turns off its checkpoint on close,
    so a read-write reader of a WAL database never folds the WAL into it.

Reads are allowed: a suite may read the held packages, the skill library,
the token embeddings. The refusal raises PermissionError (the code under
test sees a read-only disk) and the log row makes run_tests.py fail the
suite, whatever the suite's own fallback made of the error. The roots are
YAMADORI_OFFLINE_GUARD_STATE (os.pathsep-separated) when set -- the guard's
own test points it at a temp tree -- else this repository's index/ and logs/.
The hook sees Python only: a non-Python child (node, a shell) that writes a
live file is caught by run_tests.py's second layer, the before/after
snapshot of the live files (scripts/run_tests.py, LIVE STATE).
"""
from __future__ import annotations

import os
import sys

STACK_PORTS = frozenset({1234, 1235, 1237, 1238, 8888, 11434}
                        | set(range(10001, 10100)))


class OfflineGuardRefused(ConnectionRefusedError):
    """A connect to the live stack from an offline suite."""


def _record(host, port, *, kind: str = "connect", op: str = "",
            path: str = "") -> None:
    import json
    import time
    import traceback
    def where(p: str) -> str:
        try:
            return os.path.relpath(p).replace(os.sep, "/")
        except ValueError:                       # another drive (Windows)
            return p
    frames = [f"{where(f.filename)}:{f.lineno} {f.name}"
              for f in traceback.extract_stack()
              if "offline_guard" not in f.filename
              and not os.path.normcase(f.filename).startswith(
                  (os.path.normcase(sys.prefix), os.path.normcase(sys.base_prefix)))]
    row = {"kind": kind, "pid": os.getpid(), "host": str(host), "port": port,
           "argv": " ".join(sys.argv)[:200], "at": time.time(),
           "stack": frames[-8:]}
    if kind == "write":
        row.update(op=op, path=where(path))
    path = os.environ.get("YAMADORI_OFFLINE_GUARD_LOG")
    if path:
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
        except OSError:
            pass
    try:
        what = (f"a write to the live state ({op} {row.get('path')})"
                if kind == "write" else f"a connect to {host}:{port}")
        sys.stderr.write(f"OFFLINE GUARD: refused {what} "
                         f"from {frames[-1] if frames else '?'}\n")
    except Exception:                                            # noqa: BLE001
        pass


def check(address) -> None:
    """Raise OfflineGuardRefused for an (host, port, ...) on a stack port."""
    if not (isinstance(address, tuple) and len(address) >= 2):
        return
    host, port = address[0], address[1]
    if isinstance(port, int) and port in STACK_PORTS:
        _record(host, port)
        raise OfflineGuardRefused(
            10061, f"OFFLINE GUARD: an offline suite may not reach the live "
                   f"stack ({host}:{port}); use a fixture or a fake on port 0")


_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(
    __file__)), "..", ".."))


def _state_roots() -> tuple[str, ...]:
    raw = os.environ.get("YAMADORI_OFFLINE_GUARD_STATE")
    roots = ([p for p in raw.split(os.pathsep) if p.strip()] if raw else
             [os.path.join(_REPO, "index"), os.path.join(_REPO, "logs")])
    return tuple(os.path.normcase(os.path.abspath(p)) for p in roots)


STATE_ROOTS = _state_roots()
_WRITE_FLAGS = (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC
                | os.O_APPEND)
_BUSY = __import__("threading").local()


class OfflineGuardWrite(PermissionError):
    """A write to the live state from an offline suite."""


def _live(path) -> str | None:
    """The normalised path when it lies under a live-state root, else None."""
    if isinstance(path, int) or path is None:
        return None                                  # an fd: opened earlier
    try:
        p = os.fsdecode(os.fspath(path))
    except TypeError:
        return None
    try:
        n = os.path.normcase(os.path.abspath(p))
    except (ValueError, OSError):
        return None
    for r in STATE_ROOTS:
        if n == r or n.startswith(r + os.sep):
            return n
    return None


def _refuse_write(what: str, path: str) -> None:
    _record(None, None, kind="write", op=what, path=path)
    raise OfflineGuardWrite(
        13, f"OFFLINE GUARD: an offline suite may not write the live state "
            f"({what} {path}); point the store's variable at a temp path "
            f"before importing (mcp/test_sessions.py's header)")


def _sqlite_target(db) -> tuple[str | None, bool]:
    """(the file a sqlite3.connect opens, read-only?)."""
    if not isinstance(db, (str, bytes, os.PathLike)):
        return None, True
    s = os.fsdecode(os.fspath(db))
    if s == ":memory:" or not s:
        return None, True
    if s.startswith("file:"):
        from urllib.parse import parse_qs, unquote, urlsplit
        u = urlsplit(s)
        q = {k: v[-1] for k, v in parse_qs(u.query).items()}
        path = unquote(u.path or u.netloc)
        if path.startswith("/") and len(path) > 2 and path[2] == ":":
            path = path[1:]                              # file:/C:/x
        ro = q.get("mode") == "ro" or q.get("immutable") == "1"
        return (None if q.get("mode") == "memory" else path), ro
    return s, False


# SQLite: a READ-WRITE connection to a live database may READ, and every
# statement that would change the file is refused at prepare time by an
# authorizer and logged; the connection never checkpoints on close (a
# read-write reader of a WAL database would otherwise fold the WAL into the
# file when it is the last connection). Temp-schema writes are allowed.
_SQL_WRITE_NAMES = (
    "SQLITE_INSERT", "SQLITE_UPDATE", "SQLITE_DELETE", "SQLITE_CREATE_TABLE",
    "SQLITE_CREATE_INDEX", "SQLITE_CREATE_TRIGGER", "SQLITE_CREATE_VIEW",
    "SQLITE_CREATE_VTABLE", "SQLITE_DROP_TABLE", "SQLITE_DROP_INDEX",
    "SQLITE_DROP_TRIGGER", "SQLITE_DROP_VIEW", "SQLITE_DROP_VTABLE",
    "SQLITE_ALTER_TABLE", "SQLITE_ANALYZE", "SQLITE_REINDEX",
    "SQLITE_ATTACH")
# PRAGMAs that change the file when given a value.
_PRAGMA_WRITES = frozenset({"journal_mode", "user_version", "application_id",
                            "schema_version", "page_size", "auto_vacuum",
                            "writable_schema", "wal_checkpoint",
                            "incremental_vacuum", "optimize"})


def _guard_connection(con, path: str) -> None:
    import sqlite3   # already imported: this is its connect's handle event
    writes = {getattr(sqlite3, n): n[7:] for n in _SQL_WRITE_NAMES
              if hasattr(sqlite3, n)}
    denied: list[int] = [0]
    creates = {getattr(sqlite3, n) for n in (
        "SQLITE_CREATE_TABLE", "SQLITE_CREATE_INDEX", "SQLITE_CREATE_TRIGGER",
        "SQLITE_CREATE_VIEW") if hasattr(sqlite3, n)}
    schema_rows = {getattr(sqlite3, n) for n in (
        "SQLITE_INSERT", "SQLITE_UPDATE", "SQLITE_DELETE")}
    probe: list = []

    def exists(name) -> bool:
        """`CREATE ... IF NOT EXISTS` of an object that is there is a no-op
        (code_search opens every package index that way): asked on a
        separate read-only connection, never the one being authorized."""
        try:
            if not probe:
                probe.append(sqlite3.connect(f"file:{path}?mode=ro",
                                             uri=True))
            return probe[0].execute(
                "SELECT 1 FROM sqlite_master WHERE name=?", (name,)
            ).fetchone() is not None
        except Exception:                                        # noqa: BLE001
            return False

    def auth(action, a1, a2, dbname, _trigger):
        what = None
        if dbname == "temp":
            return sqlite3.SQLITE_OK
        if action in schema_rows and str(a1).lower() in (
                "sqlite_master", "sqlite_schema"):
            # The schema row of a CREATE / ALTER / DROP: that statement's
            # own action is authorized on its own (below).
            return sqlite3.SQLITE_OK
        if action in creates and a1 and exists(a1):
            return sqlite3.SQLITE_OK
        if action in writes:
            what = f"{writes[action]} {a1 or ''}".strip()
        elif action == sqlite3.SQLITE_PRAGMA and a2 is not None \
                and str(a1).lower() in _PRAGMA_WRITES:
            what = f"PRAGMA {a1}={a2}"
        if what is None:
            return sqlite3.SQLITE_OK
        if denied[0] < 50:
            denied[0] += 1
            prev = getattr(_BUSY, "on", False)
            _BUSY.on = True
            try:
                _record(None, None, kind="write", op=f"sqlite {what}",
                        path=path)
            finally:
                _BUSY.on = prev
        return sqlite3.SQLITE_DENY
    try:
        con.set_authorizer(auth)
    except Exception:                                            # noqa: BLE001
        pass
    try:
        con.setconfig(sqlite3.SQLITE_DBCONFIG_NO_CKPT_ON_CLOSE, True)
    except Exception:                                            # noqa: BLE001
        pass


def _wrap_sqlite() -> None:
    """sqlite3.connect, wrapped: the audit event names the file (and marks
    a live read-write open on this thread); once the connection exists --
    its handle event fires before Connection.__init__ has run, too early
    for an authorizer -- the guard is installed on it."""
    import sqlite3
    import sqlite3.dbapi2 as dbapi2
    orig = sqlite3.connect

    def connect(*a, **kw):
        _BUSY.sqlite_live = None
        con = orig(*a, **kw)
        path = getattr(_BUSY, "sqlite_live", None)
        _BUSY.sqlite_live = None
        if path:
            _guard_connection(con, path)
        return con
    connect.__wrapped__ = orig
    connect.__doc__ = orig.__doc__
    sqlite3.connect = connect
    dbapi2.connect = connect


def _check_write(event, args) -> None:
    if event == "open":
        path, mode, flags = (tuple(args) + (None, None, None))[:3]
        writing = (isinstance(flags, int) and bool(flags & _WRITE_FLAGS)) or (
            isinstance(mode, str) and any(c in mode for c in "wax+"))
        if writing:
            p = _live(path)
            if p:
                _refuse_write(f"open({mode or flags})", p)
    elif event == "sqlite3.connect":
        path, ro = _sqlite_target(args[0] if args else None)
        p = _live(path) if path else None
        # Read-only URIs need nothing; a read-write connection gets the
        # authorizer on its handle (the next event on this thread).
        _BUSY.sqlite_live = p if (p and not ro) else None
    elif event == "os.rename":
        for a in args[:2]:
            p = _live(a)
            if p:
                _refuse_write("os.rename/replace", p)
    elif event == "os.mkdir":
        # os.makedirs(..., exist_ok=True) on a directory that is there
        # changes nothing (it raises FileExistsError and moves on).
        p = _live(args[0] if args else None)
        if p and not os.path.isdir(p):
            _refuse_write(event, p)
    elif event in ("os.remove", "os.rmdir", "os.truncate",
                   "os.utime", "os.chmod", "shutil.rmtree", "os.link",
                   "os.symlink"):
        p = _live(args[0] if args else None)
        if p:
            _refuse_write(event, p)


_WRITE_EVENTS = frozenset({"open", "sqlite3.connect", "os.rename",
                           "os.remove", "os.rmdir", "os.mkdir", "os.truncate",
                           "os.utime", "os.chmod", "shutil.rmtree", "os.link",
                           "os.symlink"})


def _hook(event, args):
    if event == "socket.connect":
        check(args[1])
    elif event in _WRITE_EVENTS and not getattr(_BUSY, "on", False):
        _BUSY.on = True                   # the log's own open re-enters
        try:
            _check_write(event, args)
        finally:
            _BUSY.on = False


def _wrap_proactor() -> None:
    if sys.platform != "win32":
        return
    try:
        from asyncio import windows_events
    except Exception:                                            # noqa: BLE001
        return
    orig = windows_events.IocpProactor.connect

    def connect(self, conn, address):
        check(address)
        return orig(self, conn, address)
    windows_events.IocpProactor.connect = connect


if os.environ.get("YAMADORI_OFFLINE_GUARD") == "1":
    sys.addaudithook(_hook)
    _wrap_proactor()
    _wrap_sqlite()
