#!/usr/bin/env python
"""THE SANDBOX: QuickJS on WASI, inside a WebAssembly runtime (wasmtime), in-process (docs/CODEMODE.md "The runner").

Operator, 2026-10-08 (verbatim): "Yes download the wasi and quickjs stuff we need to build our codemode sandbox." The
first build ran quickjs-ng (a native extension) in a child process; this one runs the quickjs-ng project's OWN WASI
build of its `qjs` binary (`qjs-wasi.wasm`, release v0.17.0, sha256 pinned below) under wasmtime (wasmtime-py, pinned).
A WebAssembly module can touch nothing the host does not hand it, so the sandbox is the module's imports:

  * NO WASI from wasmtime. This file implements the module's 23 `wasi_snapshot_preview1` imports itself, as a shim
    that grants nothing: no preopened directory (`fd_prestat_get` and every `path_*` refuse), no environment, no
    arguments but ours, no sockets (WASI preview1 has none), no files (`fd_read` answers only fd 0, `fd_write` only
    fds 1 and 2: both are IN-MEMORY pipes to the host), `poll_oneoff` never sleeps, and the clock is a counter
    (deterministic: the same script reads the same `Date.now()` and `Math.random()` every time).
  * A hard linear-memory maximum: `Store.set_limits(memory_size=256 MiB)` (Pi's figure) for the WHOLE sandbox. A
    memory bomb is QuickJS's own "out of memory" InternalError.
  * An interrupt: wasmtime epoch interruption, ticked by a thread of the run's own; the deadline is the wall limit
    (derived from the slowest inner tool's bound), so an infinite loop traps at it. One Engine per run (compiling is
    done once: the compiled module is cached as bytes and deserialised in ~2 ms), so one run's tick or cancel never
    reaches another run's deadline.
  * The tool bridge is a line protocol over the module's stdin / stdout (below). The host runs up to four calls at
    once; the module queues Promises and resolves them as results arrive.

THE LINE PROTOCOL (JSON, one per line)
  host -> module (stdin)   {"op":"init", script, names}          the first line
                           {"op":"result", id, value}           a server function's ToolResult
                           {"op":"r", v}                        the answer to a synchronous query (below)
  module -> host (stdout)  {"op":"out", kind: "text"|"console", v}   streamed as the script writes
                           {"op":"call", id, name, args}        a server function: the host runs it, in parallel
                           {"op":"q", kind, a}                  tools.search / describe / namespaces / all: the host
                                                                answers at once with {"op":"r"}
                           {"op":"done", ok, reason, error}
The module reads the next message only when its microtask queue is EMPTY (quiescence, an `os.setTimeout(..., 0)`
turn of QuickJS's own event loop), so every continuation the results enable has run before it blocks again.

The module is the quickjs-ng project's release artifact, unmodified. Fetch it with
`python mcp/codemode_sandbox.py fetch` (verifies the sha256 before writing bin/qjs-wasi.wasm).
"""
from __future__ import annotations

import hashlib
import json
import os
import queue
import struct
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

QJS_VERSION = "v0.17.0"
QJS_URL = "https://github.com/quickjs-ng/quickjs/releases/download/v0.17.0/qjs-wasi.wasm"
QJS_SHA256 = "42a732a676ec2d93488c19411e0fad283bf72658fdad746f089914b523c783b1"
QJS_BYTES = 1569050
WASM_PATH = os.environ.get("YAMADORI_QJS_WASM") or os.path.join(ROOT, "bin", "qjs-wasi.wasm")

# One epoch tick every TICK_S seconds: how late an interrupt is SEEN (a polling granularity, not a limit).
TICK_S = 0.05

BOOT = r"""
(async () => {
  const os = await import('qjs:os');
  const std = await import('qjs:std');
  const out = std.out, inp = std.in;
  const send = (o) => { out.puts(JSON.stringify(o) + "\n"); out.flush(); };
  const readMsg = () => { const l = inp.getline(); if (l === null) std.exit(3); return JSON.parse(l); };
  const init = readMsg();
  const stash = [];
  const rpc = (kind, a) => {
    send({ op: "q", kind, a });
    for (;;) { const m = readMsg(); if (m.op === "r") return m.v; stash.push(m); }
  };
  const g = globalThis;
  for (const n of ["print", "scriptArgs", "execArgv", "argv0", "navigator", "gc"]) delete g[n];
  const EXIT = { exit: true };
  const pending = new Map();
  let seq = 0;
  const fmt = (v) => {
    if (typeof v === "string") return v;
    if (v === undefined) return "undefined";
    if (v instanceof Error) return v.name + ": " + v.message;
    try { const s = JSON.stringify(v, null, 2); return s === undefined ? String(v) : s; } catch (e) { return String(v); }
  };
  const failed = (code, reason, next) => ({ ok: false, text: reason + " " + next, error: { code, reason,
    retryable: false, next } });
  const call = (name, args) => {
    let a;
    try { a = JSON.parse(JSON.stringify(args === undefined ? {} : args)); } catch (e) {
      return Promise.resolve(failed("BAD_ARGUMENTS", name + ": the arguments could not be written as JSON (" +
        String(e.message) + ").", "pass arguments that can be written as JSON."));
    }
    return new Promise((resolve) => { const id = ++seq; pending.set(id, resolve); send({ op: "call", id, name, args: a }); });
  };
  const tools = {};
  for (const n of init.names) tools[n] = (args) => call(n, args);
  tools.search = (query, opts) => {
    const r = rpc("search", { query: String(query === undefined ? "" : query), opts: opts || {} });
    if (r.length === 0) r.hint = "no function matched; tools.namespaces() lists the groups, tools.all() every function";
    return r;
  };
  tools.describe = (name) => rpc("describe", { name: String(name === undefined ? "" : name) });
  tools.namespaces = (name) => rpc("namespaces", { name: name === undefined ? "" : String(name) });
  tools.all = () => rpc("all", {});
  // A name that is not a function answers like a function does: a result that says what to do (it never throws).
  g.tools = new Proxy(Object.freeze(tools), { get(t, k) {
    if (typeof k !== "string" || k in t) return t[k];
    return () => Promise.resolve(failed("NO_SUCH_FUNCTION", "tools." + k + " is not a function of this toolbox.",
      "tools.search(\"what you need\") ranks the functions by what they answer; tools.all() lists every one."));
  } });
  g.text = (v) => { send({ op: "out", kind: "text", v: fmt(v) }); };
  const log = (...a) => { send({ op: "out", kind: "console", v: a.map(fmt).join(" ") }); };
  g.console = Object.freeze({ log, info: log, warn: log, error: log, debug: log });
  g.exit = () => { throw EXIT; };
  const AF = (async function () {}).constructor;
  let outcome = null;
  const errinfo = (e) => ({ name: String(e && e.name || "Error"), message: String(e && e.message !== undefined ?
    e.message : e).slice(0, 2000), stack: String(e && e.stack || "").slice(0, 2000) });
  const finish = (o) => {
    if (o.ok) send({ op: "done", ok: true, reason: o.exit ? "exit" : "completed" });
    else send({ op: "done", ok: false, reason: o.syntax ? "syntax_error" : (o.error && o.error.message === "out of memory"
      ? "memory" : "js_error"), error: o.error });
    std.exit(0);
  };
  let fn;
  try { fn = new AF(init.script); } catch (e) { finish({ ok: false, syntax: true, error: errinfo(e) }); }
  let p;
  try { p = fn(); } catch (e) { finish({ ok: false, error: errinfo(e) }); }
  p.then((v) => { if (v !== undefined) send({ op: "out", kind: "text", v: fmt(v) }); outcome = { ok: true }; },
         (e) => { outcome = e === EXIT ? { ok: true, exit: true } : { ok: false, error: errinfo(e) }; });
  // Runs when the microtask queue is empty (a 0 ms timer turn): settle what the host sent, or end.
  const pump = () => {
    if (outcome !== null) finish(outcome);
    if (pending.size === 0) {
      send({ op: "done", ok: false, reason: "never_settles", error: { name: "Error",
        message: "the script awaits something no tool call can settle" } });
      std.exit(0);
    }
    if (stash.length === 0) stash.push(readMsg());
    while (stash.length) {
      const m = stash.shift();
      if (m.op !== "result") continue;
      const r = pending.get(m.id); pending.delete(m.id); if (r) r(m.value);
    }
    os.setTimeout(pump, 0);
  };
  os.setTimeout(pump, 0);
})();
"""


class Killed(Exception):
    """The host ended the run (deadline, cancel, output limit): unwinds out of the module as a trap."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class Exited(Exception):
    def __init__(self, code: int):
        super().__init__(f"exit {code}")
        self.code = code


# ----------------------------------------------------------------- the module
_LOCK = threading.Lock()
_CACHE: dict = {"art": None, "sha": None}
_VERIFIED: dict = {}


_PATCHED = threading.Lock()


def _wasmtime():
    """wasmtime, with its host-function table made thread-safe. wasmtime-py keeps every host function in ONE
    global `Slab` (wasmtime/_func.py FUNCTIONS) whose allocate / deallocate (a linker defining a function; a finalizer
    freeing one when a Store is dropped, on any thread, at any time) are unsynchronised: two scripts at once
    corrupted it ("list indices must be integers or slices, not tuple", found by running six threads at the sandbox,
    2026-10-08). One reentrant lock around the table's three methods; the calls into wasm themselves stay parallel."""
    import wasmtime
    from wasmtime import _func
    slab = _func.FUNCTIONS
    if not getattr(slab, "_yamadori_locked", False):
        with _PATCHED:
            if not getattr(slab, "_yamadori_locked", False):
                lock = threading.RLock()
                for name in ("allocate", "get", "deallocate"):
                    def locked(*args, _orig=getattr(slab, name), **kw):
                        with lock:
                            return _orig(*args, **kw)
                    setattr(slab, name, locked)
                slab._yamadori_locked = True
    return wasmtime


def verify(path: str | None = None) -> tuple[bool, str]:
    """(ok, why): the module is there and is the recorded one (sha256), checked once per path and mtime."""
    path = path or WASM_PATH
    try:
        st = os.stat(path)
    except OSError:
        return False, (f"{path} is missing: python mcp/codemode_sandbox.py fetch")
    key = (path, st.st_mtime_ns, st.st_size)
    if _VERIFIED.get("key") == key:
        return _VERIFIED["ok"], _VERIFIED["why"]
    sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
    ok = sha == QJS_SHA256
    why = "qjs-wasi.wasm " + QJS_VERSION if ok else (
        f"{path} is not the recorded module (sha256 {sha[:16]}..., recorded {QJS_SHA256[:16]}...)")
    _VERIFIED.update(key=key, ok=ok, why=why)
    return ok, why


def available() -> tuple[bool, str]:
    """Can the sandbox run here? wasmtime importable and the recorded module on disk."""
    try:
        _wasmtime()
    except Exception as e:                                       # noqa: BLE001
        return False, f"wasmtime is not importable ({type(e).__name__}: {e})"[:200]
    return verify()


def _config():
    W = _wasmtime()
    c = W.Config()
    c.epoch_interruption = True
    return c


def compiled() -> bytes:
    """The module compiled once per process (wasmtime's serialised form), cached: a run deserialises it into an
    Engine of its own in about 2 ms. Measured 2026-10-08: compile 0.10 s, 3.6 MB serialised."""
    with _LOCK:
        if _CACHE["art"] is None:
            ok, why = verify()
            if not ok:
                raise RuntimeError(why)
            W = _wasmtime()
            data = open(WASM_PATH, "rb").read()
            _CACHE["art"] = W.Module(W.Engine(_config()), data).serialize()
        return _CACHE["art"]


def warm() -> None:
    """Compile in the background (the toolbox's offer calls it), so the first script does not pay for it."""
    threading.Thread(target=lambda: _safe(compiled), daemon=True, name="codemode-warm").start()


def _safe(fn):
    try:
        return fn()
    except Exception:                                            # noqa: BLE001
        return None


# ----------------------------------------------------------------- one run --
class Run:
    """One script. `on_message(dict)` is called from the module's stdout; it returns a list of reply dicts to put at
    the head of the module's stdin (sync queries) and may start work whose results arrive on `self.inbox`."""

    def __init__(self, init: dict, on_message, *, memory_bytes: int, timeout_s: float, cancelled=None):
        self.on_message = on_message
        self.memory_bytes = int(memory_bytes)
        self.timeout_s = float(timeout_s)
        self.cancelled = cancelled
        self.inbox: queue.Queue = queue.Queue()
        self.buf = bytearray((json.dumps(init) + "\n").encode("utf-8"))
        self.out_buf = bytearray()
        self.stderr = bytearray()
        self.deadline = time.monotonic() + self.timeout_s
        self.killed: str | None = None
        self.clock = 1_700_000_000_000_000_000          # a fixed instant (ns): the sandbox's whole world of time
        self.denied: set = set()

    # -- the host side of stdin
    def _check(self) -> None:
        if self.killed:
            raise Killed(self.killed)
        if self.cancelled is not None and self.cancelled():
            self.killed = "cancelled"
            raise Killed("cancelled")
        if time.monotonic() >= self.deadline:
            self.killed = "timeout"
            raise Killed("timeout")

    def _fill(self) -> None:
        """Block until the module has something to read: a result from a worker thread."""
        while not self.buf:
            self._check()
            try:
                msg = self.inbox.get(timeout=0.05)
            except queue.Empty:
                continue
            self.buf += (json.dumps(msg, separators=(",", ":")) + "\n").encode("utf-8")

    def reply(self, msg: dict) -> None:
        self.buf += (json.dumps(msg, separators=(",", ":")) + "\n").encode("utf-8")

    # -- the shim
    def link(self, W, linker, store) -> None:
        P = "wasi_snapshot_preview1"
        I32, I64 = W.ValType.i32(), W.ValType.i64()
        argv = ["qjs", "-e", BOOT]
        run = self

        def ft(params, results=(I32,)):
            return W.FuncType(list(params), list(results))

        def mem(caller):
            return caller["memory"]

        def args_sizes_get(caller, pc, pb):
            enc = [a.encode() + b"\0" for a in argv]
            m = mem(caller)
            m.write(caller, struct.pack("<I", len(enc)), pc)
            m.write(caller, struct.pack("<I", sum(len(e) for e in enc)), pb)
            return 0

        def args_get(caller, pargv, pbuf):
            m = mem(caller)
            off = pbuf
            for i, a in enumerate(argv):
                e = a.encode() + b"\0"
                m.write(caller, struct.pack("<I", off), pargv + 4 * i)
                m.write(caller, e, off)
                off += len(e)
            return 0

        def environ_sizes_get(caller, pc, pb):
            m = mem(caller)
            m.write(caller, struct.pack("<I", 0), pc)
            m.write(caller, struct.pack("<I", 0), pb)
            return 0

        def clock_time_get(caller, cid, prec, ptr):
            run.clock += 1_000_000               # +1 ms per read: monotonic, and the same every run
            mem(caller).write(caller, struct.pack("<Q", run.clock), ptr)
            return 0

        def fd_write(caller, fd, iovs, n, pw):
            m = mem(caller)
            total = 0
            for i in range(n):
                base, ln = struct.unpack("<II", m.read(caller, iovs + 8 * i, iovs + 8 * i + 8))
                data = m.read(caller, base, base + ln) if ln else b""
                total += ln
                if fd == 1:
                    run.out_buf += data
                    run._lines()
                elif fd == 2:
                    run.stderr += data
                    del run.stderr[:-2048]
                else:
                    run.denied.add("fd_write")
                    return 8                     # EBADF
            m.write(caller, struct.pack("<I", total), pw)
            return 0

        def fd_read(caller, fd, iovs, n, pr):
            if fd != 0:
                run.denied.add("fd_read")
                return 8
            run._check()
            run._fill()
            m = mem(caller)
            total = 0
            for i in range(n):
                base, ln = struct.unpack("<II", m.read(caller, iovs + 8 * i, iovs + 8 * i + 8))
                chunk = bytes(run.buf[:ln])
                del run.buf[:ln]
                if chunk:
                    m.write(caller, chunk, base)
                    total += len(chunk)
                if len(chunk) < ln:
                    break
            m.write(caller, struct.pack("<I", total), pr)
            return 0

        def fd_fdstat_get(caller, fd, ptr):
            if fd > 2:
                run.denied.add("fd_fdstat_get")
                return 8
            # a character device; no rights beyond read / write
            mem(caller).write(caller, struct.pack("<BxHxxxxQQ", 2, 0, 0, 0), ptr)
            return 0

        def deny(code, name):
            def f(caller, *a):
                run.denied.add(name)
                return code
            return f

        def proc_exit(caller, code):
            raise Exited(code)

        def poll_oneoff(caller, inp, outp, nsub, pn):
            """Never sleeps: a clock subscription advances the counter by its timeout and fires; an fd
            subscription (stdin) fires. The module's event loop therefore never waits on the world."""
            run._check()
            m = mem(caller)
            for i in range(nsub):
                sub = m.read(caller, inp + 48 * i, inp + 48 * i + 48)
                ud, tag = struct.unpack("<QB", sub[:9])
                if tag == 0:                     # clock: timeout u64 @24, flags u16 @40 (bit 0: absolute)
                    timeout = struct.unpack("<Q", sub[24:32])[0]
                    flags = struct.unpack("<H", sub[40:42])[0]
                    if not flags & 1:
                        run.clock += int(timeout)
                ev = struct.pack("<QHBxxxxx", ud, 0, tag).ljust(32, b"\0")
                m.write(caller, ev, outp + 32 * i)
            m.write(caller, struct.pack("<I", nsub), pn)
            return 0

        table = {
            "args_sizes_get": (ft([I32, I32]), args_sizes_get),
            "args_get": (ft([I32, I32]), args_get),
            "environ_sizes_get": (ft([I32, I32]), environ_sizes_get),
            "environ_get": (ft([I32, I32]), deny(0, "environ_get")),
            "clock_time_get": (ft([I32, I64, I32]), clock_time_get),
            "fd_close": (ft([I32]), deny(0, "fd_close")),
            "fd_fdstat_get": (ft([I32, I32]), fd_fdstat_get),
            "fd_fdstat_set_flags": (ft([I32, I32]), deny(0, "fd_fdstat_set_flags")),
            "fd_prestat_get": (ft([I32, I32]), deny(8, "fd_prestat_get")),            # EBADF: no preopens
            "fd_prestat_dir_name": (ft([I32, I32, I32]), deny(8, "fd_prestat_dir_name")),
            "fd_read": (ft([I32, I32, I32, I32]), fd_read),
            "fd_readdir": (ft([I32, I32, I32, I64, I32]), deny(8, "fd_readdir")),
            "fd_seek": (ft([I32, I64, I32, I32]), deny(70, "fd_seek")),               # ESPIPE
            "fd_write": (ft([I32, I32, I32, I32]), fd_write),
            "path_create_directory": (ft([I32, I32, I32]), deny(76, "path_create_directory")),   # ENOTCAPABLE
            "path_filestat_get": (ft([I32, I32, I32, I32, I32]), deny(76, "path_filestat_get")),
            "path_filestat_set_times": (ft([I32, I32, I32, I32, I64, I64, I32]),
                                        deny(76, "path_filestat_set_times")),
            "path_open": (ft([I32, I32, I32, I32, I32, I64, I64, I32, I32]), deny(76, "path_open")),
            "path_remove_directory": (ft([I32, I32, I32]), deny(76, "path_remove_directory")),
            "path_rename": (ft([I32, I32, I32, I32, I32, I32]), deny(76, "path_rename")),
            "path_unlink_file": (ft([I32, I32, I32]), deny(76, "path_unlink_file")),
            "poll_oneoff": (ft([I32, I32, I32, I32]), poll_oneoff),
            "proc_exit": (ft([I32], []), proc_exit),
        }
        self.table = sorted(table)
        for name, (t, f) in table.items():
            linker.define_func(P, name, t, f, access_caller=True)

    def _lines(self) -> None:
        while True:
            i = self.out_buf.find(b"\n")
            if i < 0:
                return
            line = bytes(self.out_buf[:i])
            del self.out_buf[:i + 1]
            try:
                msg = json.loads(line.decode("utf-8", errors="replace"))
            except ValueError:
                continue
            if isinstance(msg, dict):
                for r in self.on_message(msg) or []:
                    self.reply(r)

    # -- go
    def run(self) -> dict:
        """Instantiate and run to exit. {"how": "exit" | "killed" | "trap", "reason", "denied", "instantiate_ms"}."""
        W = _wasmtime()
        art = compiled()
        engine = W.Engine(_config())
        t0 = time.monotonic()
        module = W.Module.deserialize(engine, art)
        store = W.Store(engine)
        store.set_limits(memory_size=self.memory_bytes)
        ticks = max(int(self.timeout_s / TICK_S) + 1, 1)
        store.set_epoch_deadline(ticks)
        linker = W.Linker(engine)
        for imp in module.imports:
            if imp.module != "wasi_snapshot_preview1":
                return {"how": "trap", "reason": f"the module imports {imp.module}.{imp.name}: refused"}
        self.link(W, linker, store)
        instance = linker.instantiate(store, module)
        start = instance.exports(store)["_start"]
        inst_ms = round((time.monotonic() - t0) * 1000, 1)
        stop = threading.Event()
        n = {"i": 0}

        def ticker():
            while not stop.wait(TICK_S):
                engine.increment_epoch()
                n["i"] += 1
                if self.cancelled is not None and self.cancelled() and not self.killed:
                    self.killed = "cancelled"
                if self.killed or time.monotonic() >= self.deadline:
                    self.killed = self.killed or "timeout"
                    while n["i"] <= ticks + 1:           # overshoot the deadline: the module traps at its next check
                        engine.increment_epoch()
                        n["i"] += 1
                    return

        th = threading.Thread(target=ticker, daemon=True, name="codemode-epoch")
        th.start()
        res: dict = {"instantiate_ms": inst_ms}
        try:
            start(store)
            res.update(how="exit", reason="returned")
        except Exited as e:
            res.update(how="exit", reason=f"exit {e.code}")
        except Killed as e:
            res.update(how="killed", reason=e.reason)
        except Exception as e:                                   # noqa: BLE001
            # a trap: the epoch deadline (the ticker said why), an exception of ours inside a host call that
            # wasmtime re-raised as a trap, or the module's own abort
            if self.killed:
                res.update(how="killed", reason=self.killed)
            else:
                code = getattr(e, "trap_code", None)
                if code == W.TrapCode.STACK_OVERFLOW:
                    # QuickJS cannot see wasm's call stack (its own check never fires), so deep recursion ends as
                    # wasmtime's stack trap: the run is over, the host is untouched
                    res.update(how="trap", reason="stack_overflow")
                elif code in (W.TrapCode.MEMORY_OUT_OF_BOUNDS, W.TrapCode.UNREACHABLE):
                    res.update(how="trap", reason="memory_trap")
                else:
                    msg = str(e)
                    res.update(how="trap", reason=f"{type(e).__name__}: {msg.splitlines()[0][:200] if msg else ''}")
        finally:
            stop.set()
        res["denied"] = sorted(self.denied)
        res["stderr"] = bytes(self.stderr).decode("utf-8", errors="replace")[-300:]
        return res


def fetch(dest: str | None = None) -> int:
    """Download the recorded module and write it only if its sha256 is the recorded one."""
    import urllib.request
    dest = dest or WASM_PATH
    with urllib.request.urlopen(QJS_URL, timeout=120) as r:
        data = r.read()
    sha = hashlib.sha256(data).hexdigest()
    if sha != QJS_SHA256:
        print(f"refused: {QJS_URL} is sha256 {sha}, recorded {QJS_SHA256}", file=sys.stderr)
        return 1
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "wb") as f:
        f.write(data)
    print(f"wrote {dest} ({len(data)} bytes, sha256 {sha})")
    return 0


if __name__ == "__main__":
    if sys.argv[1:2] == ["fetch"]:
        sys.exit(fetch())
    print(available())
