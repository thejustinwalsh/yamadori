#!/usr/bin/env python
"""The token ledger, the price snapshot and the savings estimate, asserted.
No GPU, no network, no model server: every upstream here is a fake, every
price is a fixture, and every file lives in a temp directory.

WHAT THIS IS GATING

  token_ledger.record()   the cache split (llama-server timings, else
                          usage.prompt_tokens_details, else "unsplit"),
                          reasoning only where reported, no_usage counted,
                          rows that only ever add, off until enable()
  role_of / record_upstream   main / second brain / side call from a proxy
                          payload, the account from its slot
  model.post              deep thinking (shape role="helper") lands as the
                          second brain, everything else as internal, and
                          the marker never reaches the model server
  windows()               ALL TIME, LAST 30 DAYS, THIS WEEK from Monday
  prices                  a snapshot is the whole list, an implausible body
                          is refused, the like-for-like and the median, a
                          missing cache-read price priced as input, the
                          at-most-daily schedule on the net lane
  dash_tokens             hosted cost and saving per window with fixed
                          prices, warm excluded, electricity from the first
                          recorded day, the gap before it from the corpus
"""
from __future__ import annotations

import datetime as dt
import http.server
import json
import os
import sqlite3
import sys
import tempfile
import threading
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

_TMP = tempfile.mkdtemp(prefix="yamadori_test_tokens_")
REAL_LEDGER = os.path.abspath(os.path.join(HERE, "..", "index", "token_ledger.sqlite3"))
_REAL_BEFORE = os.path.getmtime(REAL_LEDGER) if os.path.exists(REAL_LEDGER) else None
os.environ["YAMADORI_TOKEN_LEDGER"] = os.path.join(_TMP, "never.sqlite3")
os.environ["YAMADORI_PRICES_DIR"] = os.path.join(_TMP, "prices")
os.environ["YAMADORI_JOBS_DB"] = os.path.join(_TMP, "jobs.sqlite3")
os.environ["YAMADORI_CORPUS_DB"] = os.path.join(_TMP, "corpus.sqlite3")
os.environ["YAMADORI_POWER_LEDGER"] = os.path.join(_TMP, "power.json")
os.environ["YAMADORI_SLOT_PINNING"] = "0"
os.environ.pop("YAMADORI_POWER_RATE_CENTS", None)
os.environ.pop("YAMADORI_POWER_EXTRA_WATTS", None)

import budget  # noqa: E402
import dash_tokens  # noqa: E402
import jobs  # noqa: E402
import model  # noqa: E402
import prices  # noqa: E402
import slots  # noqa: E402
import tiers  # noqa: E402
import token_ledger as tl  # noqa: E402

tiers._accepted = tiers.FALLBACK_EFFORTS          # no template fetch
_real_budgets = budget.budgets
budget.budgets = lambda pool=None: _real_budgets(pool or 147456)

_results: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> bool:
    _results.append((bool(ok), name, detail))
    return bool(ok)


def near(a, b, tol=1e-9) -> bool:
    return a is not None and b is not None and abs(a - b) <= tol


def ep(*a) -> float:
    return time.mktime(dt.datetime(*a).timetuple())


def fresh(name: str) -> str:
    p = os.path.join(_TMP, name)
    tl.enable(p)
    return p


def rows(path: str) -> list[dict]:
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in con.execute("SELECT * FROM daily ORDER BY day, role")]
    finally:
        con.close()


# ---------------------------------------------------------------------------

def test_off_until_enabled():
    tl.disable()
    check(tl.record("main", usage={"prompt_tokens": 5, "completion_tokens": 1}) is False,
          "record() is a no-op before enable()")
    check(not os.path.exists(os.environ["YAMADORI_TOKEN_LEDGER"]),
          "and writes no file")
    check(tl.record_upstream({"_slot": {}}, {"usage": {}}) is False,
          "record_upstream too")


def test_the_split():
    c = tl.counts_of({"prompt_tokens": 1000, "completion_tokens": 300},
                     {"cache_n": 900, "prompt_n": 100, "predicted_n": 300})
    check(c["prompt_cached"] == 900 and c["prompt_processed"] == 100
          and c["prompt_unsplit"] == 0 and c["completion"] == 300,
          "llama-server timings split the prompt: cache_n cached, prompt_n processed", str(c))
    c = tl.counts_of({"prompt_tokens": 1000, "completion_tokens": 50,
                      "prompt_tokens_details": {"cached_tokens": 600}})
    check(c["prompt_cached"] == 600 and c["prompt_processed"] == 400,
          "without timings, usage.prompt_tokens_details.cached_tokens splits it", str(c))
    c = tl.counts_of({"prompt_tokens": 1000, "completion_tokens": 50})
    check(c["prompt_unsplit"] == 1000 and c["prompt_cached"] == 0
          and c["prompt_processed"] == 0,
          "no split reported: counted as unsplit, never guessed into either", str(c))
    c = tl.counts_of(None, {"cache_n": 10, "prompt_n": 5, "predicted_n": 7})
    check(c["completion"] == 7, "completion falls back to timings.predicted_n", str(c))
    c = tl.counts_of({"prompt_tokens": 10, "completion_tokens": 90,
                      "completion_tokens_details": {"reasoning_tokens": 60}})
    check(c["reasoning"] == 60 and c["reasoning_reported"] == 1,
          "reasoning is recorded where the server reports it", str(c))
    c = tl.counts_of({"prompt_tokens": 10, "completion_tokens": 90})
    check(c["reasoning"] == 0 and c["reasoning_reported"] == 0,
          "and an unreported one is not counted as reported", str(c))
    c = tl.counts_of(None, None)
    check(c["no_usage"] == 1 and c["generations"] == 1,
          "a generation with no counts at all is counted as no_usage", str(c))
    rec = slots.cache_record({"cache_n": 3, "prompt_n": 4}, None, None)
    check(tl.counts_of({"completion_tokens": 1}, cache=rec)["prompt_cached"] == 3,
          "the proxy's own cache record is used as given")


def test_rows_only_add():
    p = fresh("add.sqlite3")
    t0 = ep(2026, 9, 24, 10, 0)
    tl.record("main", account="acct1", usage={"prompt_tokens": 100, "completion_tokens": 10},
              timings={"cache_n": 60, "prompt_n": 40}, when=t0)
    tl.record("main", account="acct1", usage={"prompt_tokens": 50, "completion_tokens": 5},
              timings={"cache_n": 50, "prompt_n": 0}, when=t0 + 60)
    tl.record("bogus-role", usage={"prompt_tokens": 1, "completion_tokens": 1}, when=t0)
    tl.record("main", account="acct1", usage={"prompt_tokens": 7, "completion_tokens": 1},
              timings={"cache_n": 0, "prompt_n": 7}, when=ep(2026, 9, 25, 0, 0, 1))
    r = rows(p)
    day1 = [x for x in r if x["day"] == "2026-09-24" and x["role"] == "main"]
    check(len(day1) == 1 and day1[0]["generations"] == 2
          and day1[0]["prompt_cached"] == 110 and day1[0]["prompt_processed"] == 40
          and day1[0]["completion"] == 15,
          "two generations on one day add into one row", str(day1))
    check(day1[0]["first_at"] == t0 and day1[0]["last_at"] == t0 + 60,
          "the row keeps its first and last time")
    check(any(x["role"] == "internal" and x["account"] == "" for x in r),
          "an unknown role is filed as internal", str([x["role"] for x in r]))
    check(any(x["day"] == "2026-09-25" for x in r),
          "a generation after local midnight starts the next day's row")
    h = tl.history(p)
    check(h["first_at"] == t0 and h["first_day"] == "2026-09-24" and h["days"] == 2
          and h["accounts"] == 2, "history: first_at, days and accounts seen", str(h))
    check(tl.status()["errors"] == 0, "no write failed", str(tl.status()))


def test_roles_from_proxy_payloads():
    check(tl.role_of({"_slot": {"key": "abc", "transient": False}}) == "main",
          "a conversation turn is main")
    check(tl.role_of({"_utility": {"utility": True}, "_slot": {"transient": True}})
          == "side_call", "a utility call is a side call")
    check(tl.role_of({"_utility": {"utility": True, "kind": "compaction"},
                      "_slot": {"key": "k", "transient": False}}) == "side_call",
          "a compaction rewritten onto its conversation's slot is still a side call")
    check(tl.role_of({"_role": "helper"}) == "second_brain",
          "fan-out's helper candidates are the second brain")
    check(tl.role_of({"_role": "main", "_slot": {"key": "k"}}) == "main",
          "fan-out's candidate A is main")
    p = fresh("upstream.sqlite3")
    rec = slots.cache_record({"cache_n": 1000, "prompt_n": 24}, None, None)
    ok = tl.record_upstream(
        {"_slot": {"key": "k", "transient": False, "account": "acctA"}},
        {"usage": {"prompt_tokens": 1024, "completion_tokens": 77}, "_cache": rec})
    r = rows(p)
    check(ok and len(r) == 1 and r[0]["account"] == "acctA" and r[0]["role"] == "main"
          and r[0]["prompt_cached"] == 1000 and r[0]["prompt_processed"] == 24
          and r[0]["completion"] == 77,
          "record_upstream files the proxy's done item under the slot's account",
          str(r))


SEEN: list[dict] = []


class _Upstream(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        SEEN.append(json.loads(self.rfile.read(n)))
        body = json.dumps({
            "choices": [{"finish_reason": "stop",
                         "message": {"role": "assistant", "content": "ok"}}],
            "usage": {"prompt_tokens": 500, "completion_tokens": 40},
            "timings": {"cache_n": 450, "prompt_n": 50, "predicted_n": 40}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


def test_model_post_records_the_one_door():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Upstream)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    real = model.UPSTREAM
    model.UPSTREAM = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        p = fresh("door.sqlite3")
        SEEN.clear()
        model.ask([{"role": "user", "content": "summarise"}], max_tokens=100)
        model.post(model.shape({"messages": [{"role": "user", "content": "dig"}],
                                "max_tokens": 100}, "low", role="helper"))
        r = {x["role"]: x for x in rows(p)}
        check(set(r) == {"internal", "second_brain"},
              "model.ask is internal; a helper-share body is the second brain",
              str(sorted(r)))
        check(r.get("second_brain", {}).get("prompt_cached") == 450
              and r.get("second_brain", {}).get("prompt_processed") == 50
              and r.get("internal", {}).get("completion") == 40,
              "a non-streamed response's top-level timings give the split", str(r))
        check(all(not any(k.startswith("_") for k in s) for s in SEEN) and len(SEEN) == 2,
              "the _share marker never reaches the model server",
              str([sorted(s) for s in SEEN]))
        check(model.shape({"messages": []}, "low").get("_share") is None,
              "a main-share body carries no marker (shape stays tiers.apply + model)")
    finally:
        model.UPSTREAM = real
        srv.shutdown()


def test_windows():
    ws = {w["key"]: w for w in tl.windows(dt.date(2026, 9, 23))}      # a Wednesday
    check(ws["all"]["start"] is None, "all time has no start")
    check(ws["week"]["start"] == "2026-09-21", "this week starts on Monday", ws["week"]["start"])
    check(ws["d30"]["start"] == "2026-08-25", "last 30 days is today and the 29 before",
          ws["d30"]["start"])
    check(tl.windows(dt.date(2026, 9, 21))[2]["start"] == "2026-09-21",
          "on a Monday the week starts today")
    check(tl.windows(dt.date(2026, 9, 27))[2]["start"] == "2026-09-21",
          "on a Sunday it started six days ago")
    check("MONDAY 00:00" in ws["week"]["starts"], "the week start is labelled")
    p = fresh("windows.sqlite3")
    for day, n in (((2026, 8, 1), 1), ((2026, 9, 1), 10), ((2026, 9, 22), 100)):
        tl.record("main", usage={"prompt_tokens": n, "completion_tokens": n},
                  timings={"cache_n": 0, "prompt_n": n}, when=ep(*day, 12, 0))
    tl.record("warm", timings={"cache_n": 0, "prompt_n": 5000}, when=ep(2026, 9, 22, 12, 0))
    t = {w["key"]: tl.totals(w["start"], p) for w in tl.windows(dt.date(2026, 9, 23))}
    check(t["all"]["priced"]["completion"] == 111 and t["d30"]["priced"]["completion"] == 110
          and t["week"]["priced"]["completion"] == 100,
          "each window sums only its days", str({k: v["priced"]["completion"] for k, v in t.items()}))
    check(t["week"]["total"]["prompt_processed"] == 5100
          and t["week"]["priced"]["prompt_processed"] == 100,
          "warm prefill is in the total and not in what is priced",
          str(t["week"]["total"]["prompt_processed"]))


def _listing(mid, inp, out, cache=None, overrides=False):
    p = {"prompt": str(inp), "completion": str(out)}
    if cache is not None:
        p["input_cache_read"] = str(cache)
    if overrides:
        p["overrides"] = [{"min_prompt_tokens": 256000, "prompt": str(inp * 2)}]
    return {"id": mid, "name": mid, "pricing": p, "created": 1}


def _doc(extra=()):
    filler = [_listing(f"filler/m{i}", 1e-6, 2e-6) for i in range(prices.MIN_MODELS)]
    return {"data": filler + list(extra)}


def fixture_snapshot():
    """Fixed prices. Like-for-like has NO cache-read price; three AVERAGE_SET
    members are listed, the rest are missing."""
    a = prices.AVERAGE_SET
    return {"date": "2026-09-24", "fetched_at": ep(2026, 9, 24, 9, 0),
            "source": prices.SOURCE_URL, "listings": 60,
            "response": _doc([
                _listing(prices.LIKE_FOR_LIKE["id"], 2e-7, 1.6e-6),
                _listing(a[0], 1e-6, 1e-5, cache=1e-7),
                _listing(a[1], 3e-6, 2e-5, cache=3e-7, overrides=True),
                _listing(a[2], 2e-6, 4e-5),                     # no cache price
                _listing("router/auto", -1, -1),
            ])}


def test_prices_table():
    t = prices.table(fixture_snapshot())
    lf = t["like_for_like"]
    check(lf["id"] == "qwen/qwen3.5-27b" and near(lf["input"], 2e-7)
          and near(lf["output"], 1.6e-6), "the like-for-like listing is read", str(lf))
    check(lf["cache_read_listed"] is False and near(lf["cache_read"], 2e-7),
          "no cache-read price: cached input priced at the input price, and flagged")
    av = t["average"]
    check(av["n"] == 3 and av["missing_members"] == list(prices.AVERAGE_SET[3:]),
          "missing members are listed, not dropped", str(av["missing_members"]))
    check(near(av["input"], 2e-6) and near(av["output"], 2e-5),
          "the average is the median of each price", f"{av['input']} {av['output']}")
    check(near(av["cache_read"], 3e-7) and av["cache_read_listed"] == 2,
          "a member with no cache price contributes its input price to that median",
          f"{av['cache_read']}")
    check(any(m["overrides_ignored"] for m in av["members"]),
          "a listing's long-context tier is flagged as not applied")
    check(prices._per_token("-1") is None and prices._per_token("x") is None,
          "a -1 or unparseable price is no price")
    check("error" in prices.table(None), "no snapshot is said, not zero-priced")


def test_hosted_math():
    tok = {"prompt_processed": 1_000_000, "prompt_unsplit": 500_000,
           "prompt_cached": 4_000_000, "completion": 200_000}
    h = prices.hosted_usd(tok, {"input": 2e-7, "cache_read": 5e-8, "output": 1.6e-6})
    check(near(h["parts"]["input"], 0.3) and near(h["parts"]["cached_input"], 0.2)
          and near(h["parts"]["output"], 0.32) and near(h["usd"], 0.82),
          "hosted = (processed+unsplit) x in + cached x cache + completion x out", str(h))
    check(prices.hosted_usd(tok, {"missing": True}) is None
          and prices.hosted_usd(tok, {}) is None, "no price, no figure")


def test_refresh_and_schedule():
    d = os.path.join(_TMP, "prices_rs")
    now = ep(2026, 9, 24, 3, 0)
    try:
        prices.refresh(fetch=lambda: json.dumps({"data": [_listing("a", 1, 1)]}).encode(),
                       now=now, directory=d)
        check(False, "a 1-listing body is refused")
    except ValueError:
        check(True, "a 1-listing body is refused")
    check(prices.latest_path(d) is None, "and nothing was stored")
    doc = _doc([_listing(prices.LIKE_FOR_LIKE["id"], 2e-7, 1.6e-6)])
    r = prices.refresh(fetch=lambda: json.dumps(doc).encode(), now=now, directory=d)
    snap = json.load(open(r["path"], encoding="utf-8"))
    check(r["date"] == "2026-09-24" and r["like_for_like_listed"]
          and len(snap["response"]["data"]) == len(doc["data"]) and snap["sha256"],
          "a snapshot is the WHOLE response, dated, with its hash", r["path"])
    check(prices.load_latest(d)["date"] == "2026-09-24", "load_latest reads it")

    e = os.path.join(_TMP, "prices_empty")
    check(prices.due(now, e) is True, "no snapshot and no job: due")
    jid = prices.schedule(now, e)
    j = jobs.get(jid) if jid else None
    check(j and j["queue"] == prices.QUEUE and j["lane"] == "net" and j["state"] == "queued",
          "schedule() enqueues one refresh on the net lane", str(j and j["lane"]))
    check(prices.schedule(time.time() + 3600, e) is None,
          "a job created in the last day: nothing more (a failing fetch is not retried hourly)")
    con = sqlite3.connect(os.environ["YAMADORI_JOBS_DB"])
    con.execute("UPDATE jobs SET created=? WHERE id=?", (time.time() - 90000, jid))
    con.commit()
    con.close()
    check(prices.due(time.time(), e) is True, "a day later it is due again")
    os.utime(r["path"], (time.time(), time.time()))
    check(prices.due(time.time(), d) is False, "a snapshot younger than a day: not due")
    import worker
    check(worker.HANDLERS.get(prices.QUEUE) is prices.handle_refresh,
          "the worker has a handler for the queue")


def _corpus(path, stamps):
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY, turn TEXT, "
                "ts REAL, repo TEXT, kind TEXT, name TEXT, payload TEXT)")
    for ts in stamps:
        con.execute("INSERT INTO events(turn, ts, kind) VALUES('t', ?, 'turn')", (ts,))
        con.execute("INSERT INTO events(turn, ts, kind) VALUES('t', ?, 'answer')", (ts,))
    con.commit()
    con.close()


def test_dash_payload():
    p = fresh("dash.sqlite3")
    first = ep(2026, 9, 22, 10, 0)
    tl.record("main", account="a", usage={"prompt_tokens": 1_100_000, "completion_tokens": 100_000},
              timings={"cache_n": 1_000_000, "prompt_n": 100_000}, when=first)
    tl.record("second_brain", usage={"prompt_tokens": 0, "completion_tokens": 100_000},
              timings={"cache_n": 0, "prompt_n": 0}, when=ep(2026, 9, 23, 10, 0))
    tl.record("warm", timings={"cache_n": 0, "prompt_n": 9_000_000},
              when=ep(2026, 9, 23, 11, 0))
    corpus_db = os.path.join(_TMP, "gap_corpus.sqlite3")
    _corpus(corpus_db, [ep(2026, 9, 21, 11, 0), ep(2026, 9, 21, 12, 0), first + 5])
    days = {"2026-09-20": {"seconds": 3600, "gap_seconds": 0, "gpu_wh": 100,
                           "extra_wh": 0, "wh": {"off_peak": 100}, "cents": {"off_peak": 2.0}},
            "2026-09-22": {"seconds": 86400, "gap_seconds": 60, "gpu_wh": 1000,
                           "extra_wh": 0, "wh": {"off_peak": 1000}, "cents": {"off_peak": 20.0}},
            "2026-09-23": {"seconds": 86400, "gap_seconds": 0, "gpu_wh": 1000,
                           "extra_wh": 0, "wh": {"off_peak": 1000}, "cents": {"off_peak": 30.0}}}
    d = dash_tokens.payload(now=ep(2026, 9, 23, 18, 0), ledger_path=p, power_days=days,
                            snapshot=fixture_snapshot(), corpus_db=corpus_db)
    w = {x["key"]: x for x in d["windows"]}
    check([x["key"] for x in d["windows"]] == ["all", "d30", "week"],
          "three windows, all time first")
    a = w["all"]
    check(a["priced"]["completion"] == 200_000 and a["tokens"]["prompt_processed"] == 9_100_000
          and a["priced"]["prompt_processed"] == 100_000,
          "tokens: warm in the total, out of what is priced", str(a["priced"]))
    check(near(a["electricity"]["cents"], 50.0) and a["electricity"]["from"] == "2026-09-22"
          and a["electricity"]["days_measured"] == 2,
          "electricity counts from the first recorded day, not before", str(a["electricity"]))
    # like-for-like: 100k x 2e-7 + 1M x 2e-7 (no cache price) + 200k x 1.6e-6
    lf = a["hosted"]["like_for_like"]["usd"]
    check(near(lf, 0.02 + 0.2 + 0.32), "like-for-like hosted cost with fixed prices", str(lf))
    check(near(a["saved"]["like_for_like"], lf - 0.5),
          "saved = hosted - electricity", str(a["saved"]["like_for_like"]))
    # average: medians input 2e-6, cache 3e-7, output 2e-5
    av = a["hosted"]["average"]["usd"]
    check(near(av, 100_000 * 2e-6 + 1_000_000 * 3e-7 + 200_000 * 2e-5),
          "average hosted cost with fixed prices", str(av))
    wk = w["week"]
    check(wk["start"] == "2026-09-21" and near(wk["electricity"]["cents"], 50.0),
          "this week (from Monday 21st) covers both recorded days")
    g = d["history"]["gap"]
    check(g["turns"] == 2 and near(g["from"], ep(2026, 9, 21, 11, 0)),
          "the gap: corpus client turns before the first recorded generation", str(g))
    check(d["estimate"] is True and any("DEPRECIATION" in b for b in d["basis"])
          and d["prices"]["snapshot"]["date"] == "2026-09-24",
          "labelled an estimate, excluding hardware, with the snapshot date")
    empty = dash_tokens.payload(now=ep(2026, 9, 23, 18, 0),
                                ledger_path=os.path.join(_TMP, "none.sqlite3"),
                                power_days=days, snapshot=None, corpus_db=corpus_db)
    e = empty["windows"][0]
    check(e["electricity"]["cents"] is None and e["saved"]["average"] is None
          and "error" in empty["prices"] and empty["history"]["gap"]["turns"] == 3,
          "nothing recorded and no prices: no saving is claimed, and the whole "
          "corpus is the gap", str(e["saved"]))
    tl.enable(p)
    code, ctype, body = dash_tokens.handle_get("/dash/api/tokens")
    check(code == 200 and ctype == "application/json" and "windows" in json.loads(body),
          "/dash/api/tokens answers", str(code))
    check(dash_tokens.handle_get("/dash/api/vitals") is None, "and owns no other path")


def main() -> int:
    for fn in (test_off_until_enabled, test_the_split, test_rows_only_add,
               test_roles_from_proxy_payloads, test_model_post_records_the_one_door,
               test_windows, test_prices_table, test_hosted_math,
               test_refresh_and_schedule, test_dash_payload):
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
    tl.disable()
    after = os.path.getmtime(REAL_LEDGER) if os.path.exists(REAL_LEDGER) else None
    check(after == _REAL_BEFORE, "no test wrote the real token ledger")
    passed = sum(1 for ok, _, _ in _results if ok)
    total = len(_results)
    print(f"\n{'=' * 70}\n  {passed}/{total} checks passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
