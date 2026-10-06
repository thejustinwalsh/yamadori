#!/usr/bin/env python
"""THE DASHBOARD'S HISTORY: what the stack did, recorded as it happens, for
the PERFORMANCE and JJAVA pages (mcp/dash_perf.py, mcp/dash_jjava.py) and
the Skills page's injector panel.

Operator, 2026-09-30: "I also need jev/java stats ... graphs of our stats on
it, when it is being used ... a performance metrics page all around tok/s per
model graphs of speed, both GPUs etc." Before this module nothing kept a
generation's speed, a GPU's history, a model swap or the injector's outcome
past the response that carried it (x_yamadori) or the 10-minute ring in
mcp/power.py. So the history starts at the deploy that ships this file.

WHAT IS RECORDED (numbers, ids and model names -- never text):

  generations   one row per upstream generation: when, which model, whose
                (main / side_call / second_brain / internal / decider /
                warm), the slot, prompt tokens (reused / processed), the
                prefill's ms, completion tokens and the decode rate -- from
                llama-server's own `timings` (slots.cache_record); since
                2026-10-06 also the request's effort TIER, its TRAFFIC class
                (test | client), its corpus TURN id and a COLD flag (TABLES).
                Prefill tok/s is not a column: processed * 1000 / prompt_ms,
                derived where it is read. Hooks:
                token_ledger.record_upstream (the proxy's generations) and
                token_ledger.record (model.post: the worker's, the tools
                API's, the decider's reads)
  requests      one row per proxy request (recent_turns.note, from its
                x_yamadori): model, tier, route class, the decider Turn's
                questions, ms and failure, and the skills injector's record
                (stage 1 candidates, stage 2/3 outcomes, chosen, why)
  releases      every slot release the proxy's release rule wrote
                (slots._note / lane_kept_note): "lane burst ended" included
  swaps         every main-model swap max_mode.wait_ready made: from, to,
                the load's seconds, ok, what stayed loaded
  gpu           one row per card per minute from the power sampler's own
                1 s nvidia-smi reads (mcp/power.py; the proxy only):
                utilisation mean and max, VRAM used max, watts mean,
                temperature max

OFF THE RESPONSE PATH: every hook only puts a dict on a bounded queue
(QUEUE_MAX, a memory guard that changes no record's content: a full queue
drops the row and counts it); one daemon thread writes. Never raises.

ONLY WHERE THE TOKEN LEDGER RECORDS: a hook records only when this process
called token_ledger.enable() -- the long-running services' main()s
(server.py, worker.py, tools_api.py). Test suites import these modules and
drive them with fakes; with recording on by default they would write fake
rows into the real history (token_ledger's own rule, same reason). The
database is YAMADORI_STATS_DB, else stats.sqlite3 beside the token ledger's
database (index/), so a ledger pointed at a temp path takes this with it.

RETENTION: rows older than KEEP_DAYS are pruned, at most once an hour.
KEEP_DAYS = 30 is DERIVED from what reads it: the longest window the pages
offer is 30 days (token_ledger.LAST_DAYS, the token panel's "last 30 days").
"""
from __future__ import annotations

import json
import os
import queue
import sqlite3
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

KEEP_DAYS = 30
QUEUE_MAX = 4096
BATCH = 256
PRUNE_EVERY_S = 3600

_q: "queue.Queue[tuple[str, dict]]" = queue.Queue(maxsize=QUEUE_MAX)
_lock = threading.Lock()
_thread: threading.Thread | None = None
_state = {"written": 0, "dropped": 0, "errors": 0, "last_error": None,
          "pruned_at": 0.0}
# Tests (mcp/test_dash_stats.py) force recording on, at a temp path.
_force: bool | None = None

PROCESS = os.path.basename(sys.argv[0] or "") or "python"

TABLES = {
    # tier, traffic, turn and cold (2026-10-06; added by a forward-only
    # migration, _migrate: older rows hold NULL in them):
    #   tier     the effort tier the request ran at (x_yamadori.tier: a
    #            client's side call is `minimal`), where the request is known
    #   traffic  test | client, corpus.account_traffic of the request's
    #            account, resolved on the writer thread; NULL where no account
    #            is known (the worker's jobs) or the row predates the column
    #   turn     the request's corpus turn id (corpus.new_turn): the join to
    #            `requests.turn`. Never an account or a key
    #   cold     1 for the first generation of a request that loaded its
    #            model (x_yamadori.capacity.swap) or read its model file in
    #            (x_yamadori.preread), 0 for the other generations of a
    #            proxy request, NULL where it is not known
    "generations": ("ts REAL NOT NULL, model TEXT, role TEXT, process TEXT, "
                    "slot INTEGER, prompt INTEGER, reused INTEGER, "
                    "processed INTEGER, prompt_ms REAL, completion INTEGER, "
                    "predicted_ms REAL, decode_tps REAL, tier TEXT, "
                    "traffic TEXT, turn TEXT, cold INTEGER"),
    "requests": ("ts REAL NOT NULL, model TEXT, tier TEXT, utility INTEGER, "
                 "route TEXT, rec TEXT, traffic TEXT, turn TEXT"),
    "releases": ("ts REAL NOT NULL, slot INTEGER, why TEXT, by_why TEXT, "
                 "released INTEGER, skipped TEXT, cells_before INTEGER, "
                 "ms REAL, method TEXT, process TEXT"),
    "swaps": ("ts REAL NOT NULL, from_models TEXT, to_model TEXT, "
              "load_s REAL, ok INTEGER, how TEXT, left_loaded TEXT"),
    "gpu": ("ts REAL NOT NULL, idx INTEGER, uuid TEXT, name TEXT, n INTEGER, "
            "util_avg REAL, util_max REAL, used_max INTEGER, total INTEGER, "
            "watts_avg REAL, temp_max REAL"),
}


def path() -> str:
    """The database: YAMADORI_STATS_DB, else stats.sqlite3 beside the token
    ledger's (read at call time: token_ledger.enable(path) moves both)."""
    p = os.environ.get("YAMADORI_STATS_DB")
    if p:
        return p
    try:
        import token_ledger
        base = os.path.dirname(os.path.abspath(token_ledger.DB))
    except Exception:                                            # noqa: BLE001
        base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                            "index")
    return os.path.join(base, "stats.sqlite3")


def enabled() -> bool:
    if _force is not None:
        return _force
    try:
        import token_ledger
        return bool(token_ledger.enabled())
    except Exception:                                            # noqa: BLE001
        return False


def status() -> dict:
    return {"enabled": enabled(), "db": path(), "queued": _q.qsize(),
            **{k: v for k, v in _state.items()}}


# ------------------------------------------------------------------ writing --
def _connect(p: str | None = None) -> sqlite3.Connection:
    p = p or path()
    os.makedirs(os.path.dirname(os.path.abspath(p)) or ".", exist_ok=True)
    con = sqlite3.connect(p, timeout=30)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA busy_timeout=30000")
    for name, cols in TABLES.items():
        con.execute(f"CREATE TABLE IF NOT EXISTS {name}({cols})")
        _migrate(con, name, cols)
        con.execute(f"CREATE INDEX IF NOT EXISTS {name}_ts ON {name}(ts)")
    return con


def _migrate(con: sqlite3.Connection, name: str, cols: str) -> list[str]:
    """FORWARD-ONLY: a table made before a column was declared gains it
    (ALTER TABLE ADD COLUMN); its rows read NULL there. Nothing is dropped,
    renamed or rewritten. Three services open this file, so a column another
    process added a moment ago is not an error. Returns the columns added."""
    have = {r[1] for r in con.execute(f"PRAGMA table_info({name})")}
    added = []
    for decl in cols.split(", "):
        col = decl.split()[0]
        if col in have:
            continue
        try:
            con.execute(f"ALTER TABLE {name} ADD COLUMN {decl}")
            added.append(col)
        except sqlite3.OperationalError as e:
            if "duplicate column" not in str(e).lower():
                raise
    return added


def _put(table: str, row: dict) -> None:
    if not enabled():
        return
    try:
        _q.put_nowait((table, row))
    except queue.Full:
        _state["dropped"] += 1
        return
    _ensure_thread()


def _ensure_thread() -> None:
    global _thread
    with _lock:
        if _thread is not None and _thread.is_alive():
            return
        _thread = threading.Thread(target=_loop, name="stats-store",
                                   daemon=True)
        _thread.start()


def _loop() -> None:
    while True:
        item = _q.get()
        batch = [item]
        while len(batch) < BATCH:
            try:
                batch.append(_q.get_nowait())
            except queue.Empty:
                break
        flush_batch(batch)


def flush_batch(batch: list[tuple[str, dict]], p: str | None = None) -> int:
    """Write rows now (the writer thread's body; tests call it directly).
    Returns rows written. Never raises."""
    try:
        con = _connect(p)
        try:
            seen: dict[str, str | None] = {}
            for table, row in batch:
                cols = [c.split()[0] for c in TABLES[table].split(", ")]
                if "traffic" in cols and row.get("traffic") is None \
                        and row.get("account"):
                    # resolved here, on the writer thread, never on the
                    # response path (it reads the account registry); the
                    # account itself is not a column and is not kept
                    a = str(row["account"])
                    if a not in seen:
                        seen[a] = traffic_of(a)
                    row = dict(row, traffic=seen[a])
                vals = [row.get(c) for c in cols]
                con.execute(f"INSERT INTO {table}({', '.join(cols)}) VALUES("
                            f"{', '.join('?' for _ in cols)})", vals)
            now = time.time()
            if now - _state["pruned_at"] >= PRUNE_EVERY_S:
                cut = now - KEEP_DAYS * 86400
                for table in TABLES:
                    con.execute(f"DELETE FROM {table} WHERE ts < ?", (cut,))
                _state["pruned_at"] = now
            con.commit()
        finally:
            con.close()
        _state["written"] += len(batch)
        return len(batch)
    except Exception as e:                                       # noqa: BLE001
        _state["errors"] += 1
        _state["last_error"] = f"{type(e).__name__}: {e}"[:200]
        return 0


def drain(timeout: float = 5.0) -> None:
    """Wait until the queue is empty (tests)."""
    t0 = time.time()
    while not _q.empty() and time.time() - t0 < timeout:
        time.sleep(0.02)
    time.sleep(0.05)


# -------------------------------------------------------------------- hooks --
def _num(x) -> float | None:
    try:
        return None if x is None or isinstance(x, bool) else float(x)
    except (TypeError, ValueError):
        return None


def _int(x) -> int | None:
    v = _num(x)
    return None if v is None else int(v)


def traffic_of(account: str | None) -> str | None:
    """`test` | `client` (corpus.account_traffic, which fails closed to
    `test`), or None when no account is known: the worker's own jobs and a
    row from before this column are neither."""
    if not account:
        return None
    try:
        import corpus
        return corpus.account_traffic(account)
    except Exception:                                            # noqa: BLE001
        return None


def tier_name(tier) -> str | None:
    """A tier as a name: the proxy's `_tier` dict ({"name": ...}) or a string."""
    if isinstance(tier, dict):
        tier = tier.get("name")
    return str(tier)[:16] if tier else None


def cold_why(capacity: dict | None, preread: dict | None) -> str | None:
    """Why a request's first generation runs cold, or None. Its model was
    loaded for it (x_yamadori.capacity.swap), or its model file was read into
    the OS cache for it (x_yamadori.preread: a record that was not skipped).
    The prompt then waited behind the load or ran over a cold file cache."""
    if isinstance(capacity, dict) and isinstance(capacity.get("swap"), dict):
        return "swap"
    if isinstance(preread, dict) and preread and not preread.get("skipped"):
        return "preread"
    return None


def note_context(**kw) -> None:
    """Remember, for the request on this thread (its cancel token, which the
    threads it starts share: mcp/cancel.py), what its generations are filed
    under -- tier, account, turn -- for the ones that reach `generation`
    without it (the decider's reads and the side calls model.post sends).
    Nothing off a request thread: a no-op. Never raises."""
    try:
        import cancel
        tok = cancel.current()
        if tok is None:
            return
        ctx = getattr(tok, "stats_ctx", None)
        if ctx is None:
            ctx = tok.stats_ctx = {}
        ctx.update({k: v for k, v in kw.items() if v})
    except Exception:                                            # noqa: BLE001
        pass


def _context() -> dict:
    try:
        import cancel
        tok = cancel.current()
        return dict(getattr(tok, "stats_ctx", None) or {}) if tok else {}
    except Exception:                                            # noqa: BLE001
        return {}


def generation(*, model: str | None, role: str, cache: dict | None = None,
               timings: dict | None = None, usage: dict | None = None,
               slot: int | None = None, tier=None, account: str | None = None,
               turn: str | None = None, cold: bool | None = None) -> None:
    """One upstream generation. `cache` is slots.cache_record()'s record
    (the proxy has one); else it is made from `timings` / `usage`.

    `tier`, `account`, `turn` and `cold` file it (TABLES): the proxy's own
    generations pass them from their payload; a generation sent through
    model.post inside a request takes the request's (note_context). The
    account is resolved to `traffic` by the writer thread and is not
    stored."""
    try:
        if not enabled():
            return
        ctx = _context()
        tier = tier_name(tier) or tier_name(ctx.get("tier"))
        account = account or ctx.get("account")
        turn = turn or ctx.get("turn")
        t = timings if isinstance(timings, dict) else {}
        u = usage if isinstance(usage, dict) else {}
        c = cache if isinstance(cache, dict) else None
        if c is None:
            import slots
            c = slots.cache_record(t or None, u or None, None)
        completion = _int(t.get("predicted_n"))
        if completion is None:
            completion = _int(u.get("completion_tokens"))
        predicted_ms = _num(t.get("predicted_ms"))
        dec = _num(c.get("decode_tps"))
        if dec is None and completion and predicted_ms:
            dec = round(completion * 1000.0 / predicted_ms, 2)
        _put("generations", {
            "ts": time.time(), "model": str(model or "") or None, "role": role,
            "process": PROCESS,
            "slot": _int(c.get("slot") if c.get("slot") is not None else slot),
            "prompt": _int(c.get("prompt")), "reused": _int(c.get("reused")),
            "processed": _int(c.get("processed")),
            "prompt_ms": _num(c.get("prompt_ms")), "completion": completion,
            "predicted_ms": predicted_ms, "decode_tps": dec,
            "tier": tier, "account": account or None,
            "turn": str(turn)[:32] if turn else None,
            "cold": None if cold is None else (1 if cold else 0)})
    except Exception as e:                                       # noqa: BLE001
        _state["errors"] += 1
        _state["last_error"] = f"generation: {type(e).__name__}: {e}"[:200]


def _pick(d, keys) -> dict:
    return {k: d.get(k) for k in keys if isinstance(d, dict) and k in d}


def request_record(x: dict) -> dict:
    """The part of a request's x_yamadori the pages read: numbers, ids and
    model names only."""
    cap = x.get("capacity") if isinstance(x.get("capacity"), dict) else {}
    sk = x.get("skills") if isinstance(x.get("skills"), dict) else {}
    turn = sk.get("turn") if isinstance(sk.get("turn"), dict) else None
    inj = sk.get("inject") if isinstance(sk.get("inject"), dict) else None
    rec: dict = {"cache": _pick(x.get("cache") or {}, (
        "prompt", "reused", "processed", "prompt_ms", "decode_tps",
        "model_ms", "generations", "slot")) or None}
    if cap:
        rec["capacity"] = _pick(cap, ("model", "refused", "waited_s", "holder"))
    if turn is not None:
        rel = turn.get("release") if isinstance(turn.get("release"), dict) \
            else None
        rec["decider"] = {
            "on": turn.get("on"), "kind": turn.get("kind"),
            "decisions": len(turn.get("decisions") or []),
            "ms_questions": turn.get("ms_questions"),
            "ms_total": turn.get("ms_total"),
            "failure": (_pick(turn["failure"], ("code", "retryable"))
                        if isinstance(turn.get("failure"), dict) else None),
            "release": _pick(rel, ("released", "why", "skipped", "ms"))
            if rel else None,
            "thresholds": turn.get("thresholds")}
    if inj is not None:
        g = inj.get("gate") if isinstance(inj.get("gate"), dict) else {}
        items = g.get("items") if isinstance(g.get("items"), list) else []
        ij = g.get("inject") if isinstance(g.get("inject"), dict) else None
        st1 = inj.get("stage1") if isinstance(inj.get("stage1"), dict) else {}
        rec["inject"] = {
            "model": inj.get("model"), "profile": inj.get("profile"),
            "kind": inj.get("kind"),
            "stage1_skills": len(st1.get("skills") or []),
            "stage1_items": st1.get("items"),
            "items": [_pick(i, ("need", "score", "confidence", "level",
                                "tier", "pass")) for i in items][:64],
            "shortlist": len(g.get("shortlist") or []),
            "stage3": _pick(ij, ("noul", "tier", "act", "tie"))
            if ij else None,
            "chosen": len(inj.get("chosen") or []),
            "failure": (_pick(inj["failure"], ("code", "retryable"))
                        if isinstance(inj.get("failure"), dict) else
                        (_pick(g["failure"], ("code", "retryable"))
                         if isinstance(g.get("failure"), dict) else None)),
            "why": str(inj.get("why") or "")[:160], "ms": inj.get("ms"),
            "gate_ms": g.get("ms")}
    if sk:
        rec["skills_on"] = bool(sk.get("on"))
    return rec


def request(x: dict | None, turn: str | None = None,
            account: str | None = None) -> None:
    """One proxy request, from its x_yamadori (recent_turns.note). `turn` is
    its corpus turn id (the join to `generations.turn`); `account` becomes
    `traffic` on the writer thread and is not stored."""
    try:
        if not enabled() or not isinstance(x, dict):
            return
        cap = x.get("capacity") if isinstance(x.get("capacity"), dict) else {}
        model = cap.get("model")
        if not model:
            try:
                import max_mode
                model = max_mode.MAIN
            except Exception:                                    # noqa: BLE001
                model = None
        _put("requests", {
            "ts": time.time(), "model": model, "tier": x.get("tier"),
            "utility": 1 if x.get("utility") else 0,
            "route": ((x.get("route") or {}).get("class")
                      if isinstance(x.get("route"), dict) else None),
            "rec": json.dumps(request_record(x), default=str)[:20000],
            "account": account or None,
            "turn": str(turn)[:32] if turn else None})
    except Exception as e:                                       # noqa: BLE001
        _state["errors"] += 1
        _state["last_error"] = f"request: {type(e).__name__}: {e}"[:200]


def release(rec: dict | None) -> None:
    """One slot release record (slots._note / lane_kept_note)."""
    try:
        if not enabled() or not isinstance(rec, dict):
            return
        _put("releases", {
            "ts": time.time(), "slot": _int(rec.get("slot")),
            "why": str(rec.get("why") or "")[:80] or None,
            "by_why": str(rec.get("by") or "")[:80] or None,
            "released": 1 if rec.get("released") else 0,
            "skipped": str(rec.get("skipped") or rec.get("error") or "")[:160]
            or None,
            "cells_before": _int(rec.get("cells_before")),
            "ms": _num(rec.get("ms")), "method": rec.get("method"),
            "process": PROCESS})
    except Exception as e:                                       # noqa: BLE001
        _state["errors"] += 1
        _state["last_error"] = f"release: {type(e).__name__}: {e}"[:200]


def swap(rec: dict | None) -> None:
    """One main-model swap (max_mode.wait_ready's `swap` record)."""
    try:
        if not enabled() or not isinstance(rec, dict):
            return
        _put("swaps", {
            "ts": time.time(),
            "from_models": json.dumps(list(rec.get("from") or [])),
            "to_model": rec.get("to"), "load_s": _num(rec.get("load_s")),
            "ok": 1 if rec.get("ok") else 0,
            "how": str(rec.get("how") or "")[:160] or None,
            "left_loaded": json.dumps(list(rec.get("left_loaded") or []))})
    except Exception as e:                                       # noqa: BLE001
        _state["errors"] += 1
        _state["last_error"] = f"swap: {type(e).__name__}: {e}"[:200]


# GPU minutes: the sampler's 1 s reads folded per card into the current
# minute, written when the minute changes.
_gpu_acc: dict = {"minute": None, "cards": {}}


def _flush_gpu() -> None:
    m, cards = _gpu_acc["minute"], _gpu_acc["cards"]
    for idx, a in sorted(cards.items()):
        n = a["n"]
        if not n:
            continue
        _put("gpu", {"ts": float(m), "idx": idx, "uuid": a["uuid"],
                     "name": a["name"], "n": n,
                     "util_avg": round(a["util"] / n, 2) if a["util_n"] else None,
                     "util_max": a["util_max"], "used_max": a["used_max"],
                     "total": a["total"],
                     "watts_avg": (round(a["watts"] / a["watts_n"], 2)
                                   if a["watts_n"] else None),
                     "temp_max": a["temp_max"]})


def gpu_sample(t: float, rows: list[dict]) -> None:
    """One nvidia-smi read (vitals.gpus rows) at time `t`."""
    try:
        if not enabled() or not rows:
            return
        minute = int(t // 60) * 60
        if _gpu_acc["minute"] is not None and minute != _gpu_acc["minute"]:
            _flush_gpu()
            _gpu_acc["cards"] = {}
        _gpu_acc["minute"] = minute
        for r in rows:
            idx = _int(r.get("index"))
            if idx is None:
                continue
            a = _gpu_acc["cards"].setdefault(idx, {
                "uuid": r.get("uuid"), "name": r.get("name"), "n": 0,
                "util": 0.0, "util_n": 0, "util_max": None, "used_max": None,
                "total": _int(r.get("total_mib")), "watts": 0.0, "watts_n": 0,
                "temp_max": None})
            a["n"] += 1
            u = _num(r.get("util"))
            if u is not None:
                a["util"] += u
                a["util_n"] += 1
                a["util_max"] = u if a["util_max"] is None else max(a["util_max"], u)
            used = _int(r.get("used_mib"))
            if used is not None:
                a["used_max"] = used if a["used_max"] is None else max(a["used_max"], used)
            w = _num(r.get("watts"))
            if w is not None:
                a["watts"] += w
                a["watts_n"] += 1
            tc = _num(r.get("temp_c"))
            if tc is not None:
                a["temp_max"] = tc if a["temp_max"] is None else max(a["temp_max"], tc)
    except Exception as e:                                       # noqa: BLE001
        _state["errors"] += 1
        _state["last_error"] = f"gpu: {type(e).__name__}: {e}"[:200]


# ------------------------------------------------------------------ reading --
def read(table: str, since: float, until: float | None = None,
         p: str | None = None, limit: int = 200000) -> list[dict]:
    """Rows of `table` with ts in [since, until], oldest first, read-only.
    [] when the database does not exist yet or cannot be read."""
    if table not in TABLES:
        raise KeyError(table)
    p = p or path()
    if not os.path.exists(p):
        return []
    try:
        import pathlib
        uri = pathlib.Path(p).resolve().as_uri() + "?mode=ro"
        con = sqlite3.connect(uri, uri=True, timeout=2)
        try:
            con.row_factory = sqlite3.Row
            q = f"SELECT * FROM {table} WHERE ts >= ?"
            args: list = [since]
            if until is not None:
                q += " AND ts <= ?"
                args.append(until)
            q += " ORDER BY ts LIMIT ?"
            args.append(int(limit))
            return [dict(r) for r in con.execute(q, args)]
        finally:
            con.close()
    except sqlite3.Error:
        return []
