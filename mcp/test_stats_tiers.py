#!/usr/bin/env python
"""Speed statistics per effort TIER, context bucket and traffic class
(operator, 2026-10-06: "What are we averaging on tok/s per model, are we
collecting those stats, if not we should be, or at least by 'effort' tier
since we consider different qwen variants the same model behind our proxy").
No GPU, no stack port; every store is a temp path.

  migration   an old stats database (before the tier, traffic, turn and cold
              columns) gains them by ALTER TABLE ADD COLUMN, keeps its rows
              (NULL in the new columns), and takes new rows; running it again
              changes nothing
  recording   a generation is filed under the request's tier, its account's
              traffic class (resolved on the writer thread, the account never
              stored), its corpus turn id and a cold flag -- from the proxy's
              payload (token_ledger.record_upstream), from the request's
              context for a generation model.post sends (note_context), and
              through a whole fake turn (proxy.complete against a real
              http.server) where `requests` and `generations` join on `turn`
  perf API    dash_perf `by_tier` on a fixture database: p50 / mean / p90 / n
              of decode and of prefill (warm and cold apart, >= 2,048
              processed tokens only), per tier x model x role and per
              context bucket; the traffic filter; the old fields untouched
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import offline_stores  # noqa: E402

_TMP = offline_stores.isolate("yamadori_test_stats_tiers_")
os.environ.pop("YAMADORI_IMAGEGEN_URL", None)
os.environ.pop("YAMADORI_SLOTS", None)
os.environ.pop("YAMADORI_SLOT_PINNING", None)
os.environ["LLAMA_STACK_URL"] = "http://127.0.0.1:9"
os.environ["YAMADORI_MODEL_SERVER"] = "http://127.0.0.1:9"
os.environ["YAMADORI_MAX_STATE"] = os.path.join(_TMP, "max_mode.json")
os.environ["YAMADORI_OCTO_DIR"] = os.path.join(_TMP, "octo")

import budget  # noqa: E402
import tiers  # noqa: E402
import domains  # noqa: E402
import skill_select  # noqa: E402
import corpus  # noqa: E402
import proxy  # noqa: E402
import cancel  # noqa: E402
import token_ledger  # noqa: E402
import stats_store  # noqa: E402
import dash_perf  # noqa: E402
import model as _model  # noqa: E402

# the writer thread is the suite's own call (_flush); nothing writes behind it
stats_store._ensure_thread = lambda: None
budget._POOL = 163840
tiers._accepted = ("low", "medium", "xhigh")
domains.held_sources = lambda store=None: {}
skill_select.attach = lambda augmented, messages, sel, body=None: (
    augmented, {"on": False, "ids": [], "versions": [], "names": [],
                "chars": 0, "why": "stub"})

# accounts: whatever starts with "t-" is the live suite's, never the registry
corpus.account_traffic = lambda a: ("test" if str(a or "").startswith("t-")
                                    else "client")

_results: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> bool:
    _results.append((bool(ok), name, str(detail)[:700]))
    return bool(ok)


def _flush() -> int:
    batch = []
    while not stats_store._q.empty():
        batch.append(stats_store._q.get_nowait())
    return stats_store.flush_batch(batch) if batch else 0


def _cols(p: str, table: str) -> list[str]:
    con = sqlite3.connect(p)
    try:
        return [r[1] for r in con.execute(f"PRAGMA table_info({table})")]
    finally:
        con.close()


NEW_GEN = ["tier", "traffic", "turn", "cold"]
NEW_REQ = ["traffic", "turn"]


# ----------------------------------------------------------- migration ------
def test_migration():
    p = os.path.join(_TMP, "old_stats.sqlite3")
    con = sqlite3.connect(p)
    # the schema as shipped 2026-09-30, before the four columns
    con.execute("CREATE TABLE generations(ts REAL NOT NULL, model TEXT, role TEXT, "
                "process TEXT, slot INTEGER, prompt INTEGER, reused INTEGER, "
                "processed INTEGER, prompt_ms REAL, completion INTEGER, "
                "predicted_ms REAL, decode_tps REAL)")
    con.execute("CREATE TABLE requests(ts REAL NOT NULL, model TEXT, tier TEXT, "
                "utility INTEGER, route TEXT, rec TEXT)")
    t_old = time.time() - 3600       # recent: the writer prunes rows past KEEP_DAYS
    con.execute("INSERT INTO generations VALUES(?, 'bonsai', 'main', 'proxy', 0, "
                "9000, 8000, 1000, 500.0, 120, 2000.0, 60.0)", (t_old,))
    con.execute("INSERT INTO requests VALUES(?, 'bonsai', 'high', 0, 'prose', '{}')",
                (t_old,))
    con.commit()
    con.close()
    check(not set(NEW_GEN) & set(_cols(p, "generations")),
          "the fixture is the old schema")
    c = stats_store._connect(p)
    c.close()
    g, r = _cols(p, "generations"), _cols(p, "requests")
    check(set(NEW_GEN) <= set(g) and set(NEW_REQ) <= set(r),
          "an old database gains the tier, traffic, turn and cold columns",
          f"{g} / {r}")
    old = stats_store.read("generations", 0, p=p)
    check(len(old) == 1 and old[0]["decode_tps"] == 60.0 and old[0]["tier"] is None
          and old[0]["traffic"] is None and old[0]["turn"] is None
          and old[0]["cold"] is None,
          "the old row reads as it was, NULL in the new columns", json.dumps(old))
    oldr = stats_store.read("requests", 0, p=p)
    check(len(oldr) == 1 and oldr[0]["tier"] == "high" and oldr[0]["turn"] is None,
          "an old request row reads too", json.dumps(oldr))
    n = stats_store.flush_batch([("generations", {
        "ts": t_old + 60, "model": "flash-next", "role": "main", "tier": "max",
        "traffic": "client", "turn": "abc", "cold": 1, "decode_tps": 28.0})], p)
    both = stats_store.read("generations", 0, p=p)
    check(n == 1 and len(both) == 2 and both[1]["tier"] == "max" and both[1]["cold"] == 1,
          "a new row lands beside the old one", json.dumps(both[1:]))
    c = stats_store._connect(p)
    again = [stats_store._migrate(c, t, cols) for t, cols in stats_store.TABLES.items()]
    c.close()
    check(all(a == [] for a in again) and len(_cols(p, "generations")) == len(g),
          "migrating again adds nothing")
    # a read-only reader that never migrated (the dashboard on an old file)
    p2 = os.path.join(_TMP, "old_unmigrated.sqlite3")
    con = sqlite3.connect(p2)
    con.execute("CREATE TABLE generations(ts REAL NOT NULL, model TEXT, role TEXT, "
                "process TEXT, slot INTEGER, prompt INTEGER, reused INTEGER, "
                "processed INTEGER, prompt_ms REAL, completion INTEGER, "
                "predicted_ms REAL, decode_tps REAL)")
    con.execute("INSERT INTO generations VALUES(?, 'bonsai', 'main', 'proxy', 0, "
                "9000, 8000, 1000, 500.0, 120, 2000.0, 60.0)", (t_old,))
    con.commit()
    con.close()
    rows = stats_store.read("generations", 0, p=p2)
    rep = dash_perf.tier_section(rows, "all")
    check(len(rows) == 1 and rep["rows"][0]["tier"] is None
          and rep["rows"][0]["decode"]["n"] == 1,
          "the page reads an unmigrated old file: tier null, the rate kept",
          json.dumps(rep["rows"]))


# ----------------------------------------------------------- recording ------
def test_hook_rows():
    stats_store._force = True
    stats_store.generation(model="flash-next", role="main",
                           cache={"prompt": 12000, "reused": 9000, "processed": 3000,
                                  "prompt_ms": 2000.0, "decode_tps": 29.0},
                           usage={"completion_tokens": 200},
                           tier={"name": "max"}, account="t-acct", turn="turn0001",
                           cold=True)
    stats_store.generation(model="bonsai", role="main",
                           cache={"prompt": 9000, "processed": 100, "prompt_ms": 50.0,
                                  "decode_tps": 66.0},
                           usage={"completion_tokens": 90},
                           tier="medium", account="real-user", turn="turn0002",
                           cold=False)
    stats_store.generation(model="bonsai-a4000", role="internal",
                           cache={"prompt": 500, "processed": 500, "prompt_ms": 80.0,
                                  "decode_tps": 44.0}, usage={"completion_tokens": 70})
    stats_store.request({"tier": "max", "utility": False,
                         "capacity": {"model": "flash-next"}},
                        turn="turn0001", account="t-acct")
    _flush()
    stats_store._force = None
    g = stats_store.read("generations", 0, p=None)
    g = [r for r in g if r["turn"] in ("turn0001", "turn0002") or r["role"] == "internal"]
    a, b, c = g[-3], g[-2], g[-1]
    check(a["tier"] == "max" and a["traffic"] == "test" and a["turn"] == "turn0001"
          and a["cold"] == 1,
          "a generation lands with its tier, its account's traffic, its turn and cold",
          json.dumps(a))
    check(b["tier"] == "medium" and b["traffic"] == "client" and b["cold"] == 0,
          "a warm client generation: traffic client, cold 0", json.dumps(b))
    check(c["tier"] is None and c["traffic"] is None and c["turn"] is None
          and c["cold"] is None,
          "a generation with no request: tier, traffic, turn and cold unknown (NULL)",
          json.dumps(c))
    check("account" not in a and not any("t-acct" in str(v) for v in a.values()),
          "the account is resolved to a class and not stored")
    rq = [r for r in stats_store.read("requests", 0) if r["turn"] == "turn0001"]
    check(len(rq) == 1 and rq[0]["traffic"] == "test" and rq[0]["tier"] == "max",
          "a request lands with its turn id and traffic class", json.dumps(rq))
    check(stats_store.cold_why({"swap": {"to": "flash-next"}}, None) == "swap"
          and stats_store.cold_why({}, {"why": "swap", "ok": True}) == "preread"
          and stats_store.cold_why({}, {"why": "swap", "skipped": "no room"}) is None
          and stats_store.cold_why({"waited_s": 0.0}, None) is None
          and stats_store.cold_why(None, None) is None,
          "cold: a swap or a pre-read that ran; not a skipped one, not neither")


def test_request_context():
    stats_store._force = True
    stats_store.generation(model="bonsai-a4000", role="decider",
                           timings={"prompt_n": 40, "cache_n": 1000, "prompt_ms": 30.0,
                                    "predicted_n": 1, "predicted_ms": 5.0})
    tok = cancel.Token()
    with cancel.bound(tok):
        stats_store.note_context(tier="max", account="t-acct", turn="turnctx")
        seen = []

        def other_thread():                  # a thread the request starts shares its token
            with cancel.bound(tok):
                stats_store.generation(model="bonsai-a4000", role="internal",
                                       cache={"prompt": 300, "processed": 300,
                                              "prompt_ms": 40.0, "decode_tps": 40.0},
                                       usage={"completion_tokens": 50})
            seen.append(1)
        t = threading.Thread(target=other_thread)
        t.start()
        t.join()
        stats_store.generation(model="bonsai-a4000", role="decider",
                               timings={"prompt_n": 20, "cache_n": 900,
                                        "prompt_ms": 20.0, "predicted_n": 1,
                                        "predicted_ms": 4.0})
    _flush()
    stats_store._force = None
    rows = stats_store.read("generations", time.time() - 60)
    ctx = [r for r in rows if r["turn"] == "turnctx"]
    check(seen and len(ctx) == 2 and all(r["tier"] == "max" and r["traffic"] == "test"
                                         for r in ctx),
          "a generation model.post sends inside a request takes the request's "
          "tier, traffic and turn, on the request's own thread or one it started",
          json.dumps(ctx))
    bare = [r for r in rows if r["role"] == "decider" and r["turn"] is None]
    check(len(bare) >= 1 and bare[0]["tier"] is None,
          "outside a request nothing is filed", json.dumps(bare[:1]))


def test_record_upstream():
    """token_ledger.record_upstream: the proxy's payload keys -> the row."""
    token_ledger.enable(os.path.join(_TMP, "token_ledger.sqlite3"))
    try:
        payload = {"model": "flash-next", "_tier": {"name": "xhigh"},
                   "_account": "t-acct", "_turn": "turnup01",
                   "_cold": {"why": "swap", "used": False}}
        resp = {"usage": {"completion_tokens": 100},
                "_cache": {"prompt": 20000, "reused": 0, "processed": 20000,
                           "prompt_ms": 100000.0, "decode_tps": 28.0}}
        token_ledger.record_upstream(payload, resp)
        token_ledger.record_upstream(payload, resp)        # the turn's second hop
        token_ledger.record_upstream({"model": "bonsai", "_tier": {"name": "low"}},
                                     resp)                  # a warm: no turn, no flag
        _flush()
    finally:
        token_ledger.disable()
    rows = [r for r in stats_store.read("generations", time.time() - 60)
            if r["model"] in ("flash-next", "bonsai") and r["tier"] in ("xhigh", "low")]
    first, second, third = rows[-3:]
    check(first["cold"] == 1 and second["cold"] == 0 and third["cold"] is None,
          "only a request's first generation is cold; a payload with no flag is unknown",
          json.dumps([r["cold"] for r in rows[-3:]]))
    check(first["tier"] == "xhigh" and first["traffic"] == "test"
          and first["turn"] == "turnup01" and third["traffic"] is None,
          "the proxy's payload files the row: tier, traffic from the account, turn",
          json.dumps(first))


# ----------------------------------------------- a whole fake turn -----------
_script: list[dict] = []
_seen: list[dict] = []


def _sse(r: dict) -> bytes:
    def ev(delta=None, finish=None):
        return b"data: " + json.dumps({
            "id": "up-1", "object": "chat.completion.chunk", "model": "bonsai",
            "created": 1, "choices": [{"index": 0, "delta": delta or {},
                                       "finish_reason": finish}]}).encode() + b"\n\n"
    out = [ev({"role": "assistant"}), ev({"content": r["content"]}), ev({}, "stop")]
    last = {"id": "up-1", "object": "chat.completion.chunk", "choices": [],
            "usage": {"prompt_tokens": 3000, "completion_tokens": 100,
                      "total_tokens": 3100},
            "timings": {"cache_n": 500, "prompt_n": 2500, "prompt_ms": 5000.0,
                        "predicted_n": 100, "predicted_ms": 2000.0}}
    out.append(b"data: " + json.dumps(last).encode() + b"\n\n")
    out.append(b"data: [DONE]\n\n")
    return b"".join(out)


class _Up(BaseHTTPRequestHandler):
    def do_POST(self):                                           # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path.startswith("/upstream/"):
            data = json.dumps({"prompt": "".join(
                json.dumps(m, sort_keys=True) + "<|im_end|>"
                for m in body.get("messages") or [])}
                if self.path.endswith("apply-template") else {}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        _seen.append(body)
        data = _sse(_script.pop(0) if _script else {"content": "ok"})
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):                                   # noqa: D102
        pass


_srv = ThreadingHTTPServer(("127.0.0.1", 0), _Up)
threading.Thread(target=_srv.serve_forever, daemon=True).start()
proxy.UPSTREAM = f"http://127.0.0.1:{_srv.server_address[1]}"
_model.UPSTREAM = proxy.UPSTREAM


def _corpus_turns() -> list[str]:
    con = sqlite3.connect(corpus.CORPUS_DB)
    try:
        return [r[0] for r in con.execute(
            "SELECT turn FROM events WHERE kind='turn' ORDER BY id")]
    except sqlite3.OperationalError:          # no request logged yet
        return []
    finally:
        con.close()


def test_a_fake_turn():
    token_ledger.enable(os.path.join(_TMP, "token_ledger.sqlite3"))
    stats_store._force = None
    try:
        out = {}
        for label, account, effort, extra in (
                ("real, warm", "real-user", "xhigh", {}),
                ("test, cold", "t-soak", "low",
                 {"_capacity": {"model": "bonsai", "swap": {"to": "bonsai", "load_s": 60.0}}})):
            _script[:] = [{"content": "done"}]
            before = set(_corpus_turns())
            body = {"model": "yamadori", "_account": account,
                    "reasoning_effort": effort,
                    "messages": [{"role": "user", "content": f"say done ({label})"}],
                    **extra}
            d = proxy.complete(body)
            new = [t for t in _corpus_turns() if t not in before]
            out[label] = (d, new)
        _flush()
    finally:
        token_ledger.disable()
    rows = stats_store.read("generations", time.time() - 120)
    reqs = stats_store.read("requests", time.time() - 120)
    for label, tr, cold, tier in (("real, warm", "client", 0, "xhigh"),
                                  ("test, cold", "test", 1, "low")):
        d, new = out[label]
        turn = new[-1] if new else None
        mine = [r for r in rows if r["turn"] == turn]
        check(turn is not None and len(mine) == 1,
              f"{label}: one generation is filed under the request's corpus turn id",
              f"turn {turn}; new turns {new}; rows {json.dumps(mine)}")
        if not mine:
            continue
        g = mine[0]
        check(g["tier"] == tier and g["traffic"] == tr and g["cold"] == cold
              and g["role"] == "main" and g["model"] and g["decode_tps"] == 50.0,
              f"{label}: tier {tier}, traffic {tr}, cold {cold}, the decode rate as "
              "llama-server reported it", json.dumps(g))
        rq = [r for r in reqs if r["turn"] == turn]
        check(len(rq) == 1 and rq[0]["tier"] == tier and rq[0]["traffic"] == tr,
              f"{label}: its request row joins on the same turn id", json.dumps(rq))
        check(g["prompt"] == 3000 and g["processed"] == 2500
              and abs(dash_perf.prompt_tps(g) - 500.0) < 0.01,
              f"{label}: prefill tok/s is derived from processed and prompt_ms (500)",
              json.dumps(g))
    blob = json.dumps(rows + reqs)
    check("real-user" not in blob and "t-soak" not in blob and "say done" not in blob,
          "no account, key or prompt text reaches the rows")


# ------------------------------------------------------------ perf API -------
def _row(**kw) -> dict:
    base = {"ts": time.time() - 60, "model": "flash-next", "role": "main",
            "process": "proxy", "slot": 0, "prompt": 5000, "reused": 4000,
            "processed": 1000, "prompt_ms": 1000.0, "completion": 100,
            "predicted_ms": 3000.0, "decode_tps": 30.0, "tier": "max",
            "traffic": "client", "turn": None, "cold": 0}
    base.update(kw)
    return base


def test_perf_by_tier():
    p = os.path.join(_TMP, "perf_fixture.sqlite3")
    os.environ["YAMADORI_STATS_DB"] = p
    rows = []
    # flash-next, tier max, client: decode 28/30/32, one prefill pair (warm
    # and cold) at depth, a short prefill below 2,048 (no prefill rate)
    for tps, prompt in ((28.0, 5000), (30.0, 8192), (32.0, 8193), (24.0, 40000),
                        (20.0, 65536), (18.0, 65537), (16.0, 120000)):
        rows.append(_row(decode_tps=tps, prompt=prompt, processed=100))
    rows.append(_row(prompt=30000, processed=20000, prompt_ms=100000.0, cold=0))     # 200 tok/s warm
    rows.append(_row(prompt=30000, processed=20000, prompt_ms=200000.0, cold=1))     # 100 tok/s cold
    rows.append(_row(prompt=9000, processed=2047, prompt_ms=1000.0, cold=0))         # below 2,048: none
    rows.append(_row(prompt=9000, processed=2048, prompt_ms=1000.0, cold=0))         # 2,048: counts
    # bonsai medium client; a test row; a decider read; a warm; no tier / traffic
    rows.append(_row(model="bonsai", tier="medium", decode_tps=68.0, prompt=3000, processed=50))
    rows.append(_row(model="bonsai", tier="medium", decode_tps=70.0, prompt=3000, processed=50,
                     traffic="test"))
    rows.append(_row(model="bonsai", tier="medium", role="decider", completion=1,
                     decode_tps=200.0))
    rows.append(_row(model="bonsai-a4000", role="internal", tier=None, traffic=None,
                     decode_tps=44.0, cold=None))
    rows.append(_row(model="bonsai", tier=None, traffic=None, decode_tps=60.0, cold=None))
    rows.append(_row(model="bonsai", tier="medium", decode_tps=0.0, completion=1))   # one token: no rate
    stats_store.flush_batch([("generations", r) for r in rows], p)
    dash_perf._cache.clear()
    now = time.time()
    d = dash_perf.overview("24h", now)                       # default traffic: client
    bt = d["by_tier"]
    check(all(k in d for k in ("models", "gpus", "swaps", "gates", "window", "ctx_bins",
                               "sources", "left_out", "at")),
          "every old field is still there beside by_tier")
    check(bt["traffic"] == "client" and bt["window"] == "24h"
          and bt["ctx_buckets"] == ["0-8K", "8-32K", "32-64K", "64K+"]
          and bt["prefill_min_processed"] == 2048,
          "the default is client traffic over the 24h window; the display buckets are named",
          json.dumps({k: bt[k] for k in ("traffic", "window", "ctx_buckets")}))
    fm = next((r for r in bt["rows"] if r["tier"] == "max" and r["model"] == "flash-next"), {})
    dec = fm.get("decode", {})
    # eleven decode rates: 28 30 32 24 20 18 16 and four rows of 30
    # (sorted 16 18 20 24 28 30 30 30 30 30 32: sum 288)
    check(dec.get("n") == 11 and dec.get("p50") == 30.0 and dec.get("p90") == 30.0
          and dec.get("mean") == 26.18,
          "decode n, p50, mean and p90 for tier max on flash-next, from the rows",
          json.dumps(dec))
    # warm prefill: 20,000 tokens in 100 s = 200, 2,048 in 1 s = 2,048; the
    # 2,047-token row has none; cold: 20,000 in 200 s = 100
    pw, pc = fm.get("prefill_warm", {}), fm.get("prefill_cold", {})
    check(pw.get("n") == 2 and pw["p50"] == 200.0 and pw["mean"] == 1124.0
          and pw["p90"] == 2048.0 and pc.get("n") == 1 and pc["p50"] == 100.0,
          "prefill warm and cold apart; below 2,048 processed tokens there is no "
          "prefill rate", json.dumps({"warm": pw, "cold": pc}))
    # edges: 8192 -> 0-8K, 8193 -> 8-32K, 65536 -> 32-64K, 65537 -> 64K+
    check(dash_perf.tier_ctx(8192) == "0-8K" and dash_perf.tier_ctx(8193) == "8-32K"
          and dash_perf.tier_ctx(32768) == "8-32K" and dash_perf.tier_ctx(32769) == "32-64K"
          and dash_perf.tier_ctx(65536) == "32-64K" and dash_perf.tier_ctx(65537) == "64K+"
          and dash_perf.tier_ctx(None) is None,
          "the display buckets' edges")
    ctx = {c["bucket"]: c for c in fm["by_ctx"]}
    check(list(ctx) == ["0-8K", "8-32K", "32-64K", "64K+"]
          and [ctx[k]["decode"]["n"] for k in ctx] == [2, 5, 2, 2]
          and ctx["0-8K"]["decode"]["p50"] == 28.0
          and ctx["64K+"]["decode"]["p50"] == 16.0,
          "decode by context bucket, in order, from the rows' prompt tokens",
          json.dumps({k: v["decode"] for k, v in ctx.items()}))
    b8 = ctx["8-32K"]
    check(b8["prefill_warm"]["n"] == 2 and b8["prefill_cold"]["n"] == 1
          and ctx["0-8K"]["prefill_warm"]["n"] == 0,
          "prefill warm / cold by bucket", json.dumps(b8))
    bm = next((r for r in bt["rows"] if r["tier"] == "medium" and r["model"] == "bonsai"), {})
    check(bm.get("decode", {}).get("n") == 1 and bm["decode"]["p50"] == 68.0,
          "client traffic leaves out the test row, the decider read and a one-token "
          "generation", json.dumps(bm.get("decode")))
    check(not any(r["tier"] is None for r in bt["rows"])
          and not any(r["role"] == "decider" for r in bt["rows"]),
          "no unrecorded tier under client traffic; a decider read is never a row")
    check(bt["generations"]["client"] + bt["generations"]["test"]
          + bt["generations"]["unrecorded"] == bt["generations"]["in_window"]
          and bt["generations"]["test"] == 1 and bt["generations"]["unrecorded"] == 2,
          "the window's rows counted by traffic class", json.dumps(bt["generations"]))
    # all, test
    dall = dash_perf.overview("24h", now, "all")["by_tier"]
    check(any(r["tier"] is None and r["model"] == "bonsai-a4000" and r["role"] == "internal"
              for r in dall["rows"])
          and any(r["tier"] is None and r["model"] == "bonsai" for r in dall["rows"])
          and next(r for r in dall["rows"] if r["tier"] == "medium"
                   and r["model"] == "bonsai")["decode"]["n"] == 2
          and dall["rows"][-1]["tier"] is None,
          "all: the test rows and the unrecorded tier (sorted last) are in",
          json.dumps([(r["tier"], r["model"], r["role"]) for r in dall["rows"]]))
    dtest = dash_perf.overview("24h", now, "test")["by_tier"]
    check([(r["tier"], r["model"]) for r in dtest["rows"]] == [("medium", "bonsai")]
          and dtest["rows"][0]["decode"]["p50"] == 70.0,
          "test: only the test accounts' rows", json.dumps(dtest["rows"])[:300])
    mods = {(m["model"], m["role"]): m for m in bt["by_model"]}
    check(mods[("flash-next", "main")]["decode"]["n"] == 11 and ("bonsai", "main") in mods,
          "a per-model rollup over the tiers", json.dumps(sorted(mods)))
    # the path forms
    code, ctype, raw = dash_perf.handle_get("/dash/api/perf/7d/all")
    j = json.loads(raw)
    check(code == 200 and j["by_tier"]["traffic"] == "all" and j["by_tier"]["window"] == "7d",
          "/dash/api/perf/<window>/<traffic>", f"{code}")
    code, _, raw = dash_perf.handle_get("/dash/api/perf/24h?traffic=test")
    check(code == 200 and json.loads(raw)["by_tier"]["traffic"] == "test",
          "?traffic= on the path is read too")
    code, _, raw = dash_perf.handle_get("/dash/api/perf")
    check(code == 200 and json.loads(raw)["by_tier"]["traffic"] == "client",
          "/dash/api/perf is client traffic")
    code, _, raw = dash_perf.handle_get("/dash/api/perf/24h/nobody")
    check(code == 404 and "client" in json.loads(raw)["error"],
          "an unknown traffic class is a 404 that names the choices")
    check(dash_perf.handle_get("/dash/api/perf/24h/all/x") is None
          and dash_perf.handle_get("/dash/api/perf/9y")[0] == 404,
          "a longer path is not ours; an unknown window is 404")
    dash_perf._cache.clear()
    os.environ.pop("YAMADORI_STATS_DB", None)


def main() -> int:
    for fn in (test_migration, test_hook_rows, test_request_context,
               test_record_upstream, test_a_fake_turn, test_perf_by_tier):
        print(f"\n--- {fn.__name__} ---")
        n0 = len(_results)
        try:
            fn()
        except Exception:                                        # noqa: BLE001
            check(False, f"{fn.__name__} itself raised",
                  traceback.format_exc().strip().split("\n")[-3:])
        for ok, name, detail in _results[n0:]:
            print(("  pass  " if ok else "  FAIL  ") + name
                  + (f"   <- {detail}" if not ok and detail else ""))
    passed = sum(1 for ok, _, _ in _results if ok)
    total = len(_results)
    print(f"\n{'=' * 70}\n  {passed}/{total} checks passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
