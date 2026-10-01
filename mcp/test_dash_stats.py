#!/usr/bin/env python
"""The dashboard's history (mcp/stats_store.py) and the two pages over it
(mcp/dash_jjava.py JJAVA, mcp/dash_perf.py PERFORMANCE), on fixtures.

  stats_store   off unless the token ledger records (a suite writes nothing
                by importing); on, each hook's row lands with numbers only;
                GPU seconds fold into one row per card per minute; a full
                queue drops and counts, never blocks
  dash_jjava    decisions per bucket by question set / model / caller, the
                Jev calls counted as their questions; latency per read and
                per burst; answer labels (a stop judge's option, "none" vs
                "an option" for a skill choice); confidence and noul
                histograms; the tiers fired; the Jev routes' statuses, models
                asked for (aliases included), usage; the injector per model
  dash_perf     tok/s per model per bucket and per context bin (the gates'
                depths), decider reads apart; GPU minutes folded per window;
                swaps; every run list of a gate file and a kv_rank file with
                its n
Every store is a temp path; nothing reaches a port.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import offline_stores  # noqa: E402

_TMP = offline_stores.isolate("yamadori_test_dash_stats_")
os.environ["YAMADORI_OCTO_DIR"] = os.path.join(_TMP, "octo")
# llama-swap's /running (the Jev section's card, max_mode) and the model's
# own port answer nothing here: a reserved port that refuses.
os.environ["LLAMA_STACK_URL"] = "http://127.0.0.1:9"
os.environ["YAMADORI_MODEL_SERVER"] = "http://127.0.0.1:9"
os.environ["YAMADORI_MAX_STATE"] = os.path.join(_TMP, "max_mode.json")
os.environ["YAMADORI_DECIDER_MODELS_DIR"] = tempfile.mkdtemp(prefix="dm_", dir=_TMP)

import stats_store  # noqa: E402

# No writer thread: each test writes what the hooks queued itself (_flush).
stats_store._ensure_thread = lambda: None

_results: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> None:
    _results.append((bool(ok), name, str(detail)[:600]))


NOW = time.time()


def _flush() -> None:
    """Write whatever the hooks queued, synchronously (no writer thread)."""
    batch = []
    while not stats_store._q.empty():
        batch.append(stats_store._q.get_nowait())
    if batch:
        stats_store.flush_batch(batch)


def test_off_unless_the_ledger_records():
    stats_store._force = None
    import token_ledger
    check(not token_ledger.enabled() and not stats_store.enabled(),
          "off in a suite: the token ledger does not record here")
    stats_store.generation(model="bonsai", role="main",
                           cache={"prompt": 10, "processed": 10, "prompt_ms": 5.0,
                                  "decode_tps": 30.0})
    check(stats_store._q.empty(), "an off hook queues nothing")
    check(os.path.abspath(stats_store.path()).startswith(os.path.abspath(_TMP)),
          "the database is the suite's temp path", stats_store.path())


def test_hooks_write_rows():
    stats_store._force = True
    stats_store.generation(model="bonsai", role="main",
                           cache={"prompt": 9000, "reused": 8000, "processed": 1000,
                                  "prompt_ms": 1000.0, "decode_tps": 55.5, "slot": 0},
                           usage={"completion_tokens": 120})
    stats_store.generation(model="flash-next", role="decider",
                           timings={"prompt_n": 40, "cache_n": 2000, "prompt_ms": 80.0,
                                    "predicted_n": 1, "predicted_ms": 20.0})
    stats_store.release({"slot": 2, "why": "lane burst ended", "by": "decider batch",
                         "released": True, "cells_before": 3000, "ms": 12.0,
                         "method": "shrink"})
    stats_store.release({"slot": 2, "why": "decider batch", "released": False,
                         "skipped": "the lane is kept (layout v2: slots THE LANE)"})
    stats_store.swap({"from": ["bonsai"], "to": "flash-next", "load_s": 64.7,
                      "ok": True, "how": "HTTP 200", "left_loaded": []})
    x = {"tier": "high", "utility": False, "route": {"class": "agent_step"},
         "capacity": {"model": "bonsai", "waited_s": 0.0},
         "cache": {"prompt": 9000, "processed": 1000, "decode_tps": 55.5,
                   "prompt_ms": 1000},
         "skills": {"on": True,
                    "turn": {"on": True, "kind": "step", "decisions": ["a", "b"],
                             "ms_questions": 900.0, "ms_total": 1200.0,
                             "failure": None, "release": None},
                    "inject": {"model": "bonsai", "profile": "bonsai", "kind": "step",
                               "stage1": {"skills": ["s1", "s2"], "items": 5},
                               "gate": {"items": [
                                   {"key": "s1#0", "need": 0.8, "score": 2.7,
                                    "level": "3", "tier": "untuned", "pass": True},
                                   {"key": "s1#1", "need": 0.1, "score": 1.0,
                                    "level": "1", "tier": "untuned", "pass": False}],
                                   "shortlist": ["s1#0"],
                                   "inject": {"noul": 0.7, "tier": "untuned",
                                              "act": "all", "tie": False},
                                   "ms": 850.0},
                               "chosen": ["s1#0"], "why": "1 item(s) from 1 skill(s)",
                               "ms": 870.0,
                               "text": "NEVER STORED"}}}
    stats_store.request(x)
    t0 = 1_790_000_040                           # a minute's start
    for s in range(0, 120):                      # two minutes of 1 s reads
        stats_store.gpu_sample(t0 + s, [
            {"index": 0, "uuid": "GPU-a", "name": "RTX 5060 Ti", "util": 50 + (s % 2) * 10,
             "used_mib": 14000 + s, "total_mib": 16311, "watts": 100.0, "temp_c": 60 + s % 5},
            {"index": 1, "uuid": "GPU-b", "name": "RTX A4000", "util": 0,
             "used_mib": 600, "total_mib": 16376, "watts": 20.0, "temp_c": None}])
    stats_store.gpu_sample(t0 + 200, [{"index": 0, "util": 1}])   # flushes minute 2
    _flush()
    g = stats_store.read("generations", 0)
    check(len(g) == 2 and g[0]["model"] == "bonsai" and g[0]["decode_tps"] == 55.5
          and g[0]["completion"] == 120 and g[0]["processed"] == 1000,
          "a proxy generation lands with its speed and tokens", json.dumps(g[:1]))
    check(g[1]["role"] == "decider" and g[1]["processed"] == 40
          and g[1]["prompt_ms"] == 80.0 and g[1]["completion"] == 1,
          "a decider read lands from its timings, named", json.dumps(g[1:]))
    r = stats_store.read("releases", 0)
    check(len(r) == 2 and r[0]["why"] == "lane burst ended" and r[0]["released"] == 1
          and r[1]["skipped"].startswith("the lane is kept"),
          "releases and lane-kept notes land", json.dumps(r))
    sw = stats_store.read("swaps", 0)
    check(len(sw) == 1 and sw[0]["to_model"] == "flash-next" and sw[0]["load_s"] == 64.7,
          "a swap lands with its load seconds", json.dumps(sw))
    rq = stats_store.read("requests", 0)
    rec = json.loads(rq[0]["rec"]) if rq else {}
    check(len(rq) == 1 and rq[0]["model"] == "bonsai" and rq[0]["route"] == "agent_step"
          and rec.get("decider", {}).get("ms_questions") == 900.0
          and rec.get("inject", {}).get("stage3", {}).get("act") == "all"
          and rec.get("inject", {}).get("chosen") == 1,
          "a request lands with its decider Turn and injector record",
          json.dumps(rec)[:400])
    check("NEVER STORED" not in rq[0]["rec"], "no text of the injection is kept")
    gpu = stats_store.read("gpu", 0)
    c0 = [x for x in gpu if x["idx"] == 0]
    check(len(gpu) == 4 and len(c0) == 2 and c0[0]["n"] == 60
          and c0[0]["util_avg"] == 55.0 and c0[0]["util_max"] == 60
          and c0[0]["temp_max"] == 64 and c0[0]["used_max"] == 14059,
          "GPU seconds fold into one row per card per minute", json.dumps(gpu)[:500])
    stats_store._force = None


def test_queue_is_bounded():
    stats_store._force = True
    real = stats_store._q
    import queue
    stats_store._q = queue.Queue(maxsize=1)
    stats_store._ensure_thread = lambda: None
    try:
        d0 = stats_store._state["dropped"]
        stats_store.swap({"to": "a"})
        stats_store.swap({"to": "b"})
        check(stats_store._state["dropped"] == d0 + 1,
              "a full queue drops the row and counts it, never blocks")
    finally:
        stats_store._q = real
        stats_store._force = None


# ----------------------------------------------------------------- jjava ----
def _decisions_fixture(path: str) -> None:
    rows = []

    def row(i, name, qtype, keys, p, pick, **kw):
        r = {"row": "decision", "v": 2, "id": f"d{i}", "ts": NOW - 600 + i,
             "model": kw.pop("model", "bonsai"), "request": kw.pop("request", "u:1"),
             "kind": kw.pop("kind", "user"),
             "question": {"type": qtype, "name": name, "keys": keys},
             "orders": [{"printed": keys, "p": p}, {"printed": keys[::-1], "p": p}],
             "p": p, "pick": pick, "argmax": pick, "tie": False,
             "tier": kw.pop("tier", "untuned"), "argmax_agree": True}
        r.update(kw)
        rows.append(r)
    row(0, "build_intent", "noul", ["true", "false"], {"true": 0.9, "false": 0.1},
        "true", noul=0.9, ms=400.0, reads=2)
    row(1, "phase", "choice", ["plan", "implement", "debug", "verify"],
        {"plan": 0.1, "implement": 0.7, "debug": 0.1, "verify": 0.1}, "implement",
        confidence=0.6, ms=500.0, reads=2)
    row(2, "choose:stop", "choice", ["A", "B", "C", "D"],
        {"A": 0.1, "B": 0.7, "C": 0.1, "D": 0.1}, "B", confidence=0.6,
        option_ids=["finished", "next_step", "asking", "none"], choose_kind="stop",
        kind="stop", request="u:2", ms=450.0, reads=2)
    row(3, "choose:any", "choice", ["A", "B", "C"], {"A": 0.1, "B": 0.1, "C": 0.8},
        "C", confidence=0.7, option_ids=["h1", "h2", "none"], choose_kind="match")
    row(4, "skill_item:s1#0", "score", ["0", "1", "2", "3"],
        {"0": 0.0, "1": 0.1, "2": 0.1, "3": 0.8}, "3", confidence=0.73, score=2.7,
        model="flash-next", request="u:3", kind="step", ms=300.0, reads=2)
    row(5, "skill_inject", "noul", ["true", "false"], {"true": 0.5, "false": 0.5},
        "true", noul=0.5, model="flash-next", request="u:3", kind="step",
        tier="low", ms=320.0, reads=2)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
        f.write("not json\n")
        f.write(json.dumps({"row": "decision", "v": 2, "id": "old", "ts": NOW - 40 * 86400,
                            "question": {"name": "phase"}}) + "\n")
        f.write(json.dumps({"row": "join", "v": 2, "ts": NOW}) + "\n")


def _jev_fixture() -> None:
    import corpus
    base = {"account": "a" * 16, "traffic": "client"}
    corpus.log("r1", "jev_call", name="/jev/v1/systemone", payload=dict(
        base, route="/jev/v1/systemone", request_id="r1", status=200, ms=1200.0,
        requested="jev-latest", model="bonsai", alias_of="jjava-latest",
        questions={"q1": {"type": "noul", "options": 0},
                   "q2": {"type": "choice", "options": 3}},
        usage={"input_tokens": 900, "output_tokens": 4}))
    corpus.log("r2", "jev_call", name="/v1/systemone", payload=dict(
        base, route="/v1/systemone", request_id="r2", status=429, ms=1.0,
        requested="jjava-latest", error={"status": 429, "code": "lane_busy"}))
    corpus.log("r3", "jev_call", name="/jev/v1/systemone", payload={
        "route": "/jev/v1/systemone", "request_id": "r3", "status": 401,
        "traffic": "unknown", "ms": 0.0, "error": {"status": 401, "code": "unauthorised"}})
    corpus.log("r4", "jev_call", name="/jev/v1/models", payload=dict(
        base, route="/jev/v1/models", request_id="r4", status=200, ms=3.0, models=4))


def test_jjava_overview():
    import dash_jjava as J
    import decide_turn
    _decisions_fixture(decide_turn.DECISIONS)
    _jev_fixture()
    J._cache.clear()
    d = J.overview("24h", NOW)
    src = d["sources"]["decisions"]
    check(src["rows"] == 6 and src["bad_lines"] == 1 and src["with_latency"] == 5,
          "the decision log's v2 rows in the window are read; a bad line is counted",
          json.dumps(src))
    u = d["usage"]
    check(u["totals"].get("phase") == 1 and u["totals"].get("choose:stop") == 1
          and u["totals"].get("skill_item") == 1 and u["totals"].get("jev api") == 2,
          "decisions by question set, a Jev call counted as its questions",
          json.dumps(u["totals"]))
    callers = {k: sum(v) for k, v in u["by_caller"].items()}
    check(callers.get("skills injector") == 2 and callers.get("turn facts") == 2
          and callers.get("stop judge") == 1 and callers.get("skill selection") == 1
          and callers.get("jev /jev/v1/systemone") == 2,
          "and by caller: the injector, the proxy's other uses, the Jev route",
          json.dumps(callers))
    models = {k: sum(v) for k, v in u["by_model"].items()}
    check(models == {"bonsai": 6, "flash-next": 2}, "and by model", json.dumps(models))
    lat = d["latency"]
    check(lat["per_read"]["n"] == 5 and lat["per_read"]["p50"] == 200.0
          and lat["per_burst"]["jev_calls"]["p50"] == 1200.0
          and lat["jev_per_read"]["p50"] == 300.0,
          "latency per read (ms / reads) and per burst (a Jev call)",
          json.dumps({k: lat[k] for k in ("per_read", "per_burst", "jev_per_read")})[:400])
    qs = {q["name"]: q for q in d["question_sets"]}
    check(qs["choose:stop"]["picks"] == {"next_step": 1},
          "a stop judge's pick is its option's name", json.dumps(qs["choose:stop"]["picks"]))
    check(qs["choose"]["picks"] == {"none": 1}, "a skill choice reads none vs an option",
          json.dumps(qs["choose"]["picks"]))
    check(qs["build_intent"]["noul_hist"][9] == 1 and qs["phase"]["confidence_hist"][6] == 1,
          "confidence and noul histograms, ten bins", json.dumps(
              [qs["build_intent"]["noul_hist"], qs["phase"]["confidence_hist"]]))
    check(qs["skill_inject"]["noul_middle"] == 1,
          "a noul near 0.5 is counted as the middle band")
    check(d["thresholds"]["fired"]["skill_inject"] == {"low": 1}
          and d["thresholds"]["state"] == "untuned",
          "the tiers fired per question set, and the tables' state",
          json.dumps(d["thresholds"]["fired"]))
    jev = d["jev"]
    routes = {r["route"]: r for r in jev["routes"]}
    check(routes["/jev/v1/systemone"]["statuses"] == {"200": 1, "401": 1}
          and routes["/v1/systemone"]["statuses"] == {"429": 1}
          and routes["/jev/v1/models"]["requests"] == 1,
          "every Jev route with its statuses", json.dumps(routes)[:400])
    check(jev["requested"].get("jev-latest") == 1 and jev["served"] == {"bonsai": 1}
          and jev["usage"]["input_tokens"] == 900 and jev["errors"].get("401 unauthorised") == 1,
          "models asked for (a Jev alias), served, usage, errors", json.dumps(
              {k: jev[k] for k in ("requested", "served", "errors")}))
    check("error" not in d["priors"] and "bonsai" in d["priors"],
          "each model's priors, measured or not", json.dumps(d["priors"])[:300])


def test_jjava_injector_and_lane():
    import dash_jjava as J
    stats_store._force = True
    stats_store.request({"tier": "high", "capacity": {"model": "flash-next"},
                         "skills": {"on": True, "inject": {
                             "model": "flash-next", "stage1": {"skills": ["s"], "items": 3},
                             "gate": {"items": [], "failure": {"code": "MODEL_UNREACHABLE",
                                                               "retryable": True}},
                             "chosen": [], "failure": None,
                             "why": "decider unavailable: nothing goes in"}}})
    stats_store.release({"slot": 2, "why": "lane burst ended", "released": True})
    _flush()
    stats_store._force = None
    J._cache.clear()
    d = J.overview("24h", time.time())
    inj = {m["model"]: m for m in d["injector"]["models"]}
    b = inj.get("bonsai", {})
    check(b.get("runs") == 1 and b.get("injected") == 1 and b.get("stage3") == {"all": 1}
          and b.get("stage2_passed_mean") == 1.0 and b.get("stage1_items_mean") == 5,
          "the injector per model: runs, injected, stage 2 passed, stage 3's act",
          json.dumps(b)[:400])
    f = inj.get("flash-next", {})
    check(f.get("skipped") == 1 and f.get("failures") == {"MODEL_UNREACHABLE": 1},
          "a run jjava could not answer is skipped, with its failure code",
          json.dumps(f)[:400])
    lane = d["lane"]
    check(sum(lane["lane_burst_ended"]) >= 1 and lane["by_why"].get("lane burst ended", 0) >= 1,
          "lane releases counted per bucket", json.dumps(lane["by_why"]))


# ------------------------------------------------------------------ perf ----
def _gate_fixture() -> str:
    d = os.path.join(os.environ["YAMADORI_OCTO_DIR"], "flashnext-gate-20260929")
    os.makedirs(d, exist_ok=True)
    runs = lambda tps, n=3: [{"prompt_n": 4432, "prompt_tps": 150.0 + i, "predicted_n": 256,
                              "tps": tps + i, "n_ctx_used": 4432} for i in range(n)]
    g = {"started": "2026-09-30T07:57:28-0400", "steps": ["speed"],
         "arms": {"base": {"needles": {"hits": 15, "PASS": True},
                           "speed_eos_run": {"load_s": 19.2, "4k": runs(17.0),
                                             "32k": [{"prompt_n": 30598, "prompt_tps": 153.0,
                                                      "predicted_n": 1, "tps": 0.0}]}},
                  "flash-all": {"speed": {"4k": runs(23.0), "128k": runs(15.0)}}}}
    p = os.path.join(d, "gate.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(g, f)
    return p


def test_perf_overview():
    import dash_perf as P
    stats_store._force = True
    for i, (m, role, prompt, dec) in enumerate([
            ("bonsai", "main", 3000, 60.0), ("bonsai", "main", 40000, 40.0),
            ("bonsai", "side_call", 2000, 65.0), ("flash-next", "main", 100000, 15.0)]):
        stats_store.generation(model=m, role=role,
                               cache={"prompt": prompt, "processed": 500, "prompt_ms": 1000.0,
                                      "decode_tps": dec}, usage={"completion_tokens": 100})
    _flush()
    stats_store._force = None
    _gate_fixture()
    P._cache.clear()
    d = P.overview("24h", time.time())
    m = {x["model"]: x for x in d["models"]}
    b = m.get("bonsai", {})
    # four bonsai generations: 9000 (test_hooks_write_rows), 3000, 40000, 2000
    check(b.get("decode_tps", {}).get("n") == 4 and b["roles"].get("main") == 3
          and b["roles"].get("side_call") == 1,
          "tok/s per model, by role", json.dumps(b)[:400])
    check(m.get("flash-next", {}).get("roles", {}).get("decider") == 1
          and m["flash-next"]["decode_tps"]["n"] == 1,
          "a decider read is counted apart: no decode rate from one token",
          json.dumps(m.get("flash-next"))[:300])
    bins = {c["bin"]: c for c in b.get("by_ctx", [])}
    check(list(bins) == ["0K-4K", "8K-32K", "32K-64K"]
          and bins["32K-64K"]["decode"]["p50"] == 40.0 and bins["0K-4K"]["n"] == 2,
          "per context bin, at the gates' depths", json.dumps(b.get("by_ctx")))
    check(P.ctx_bin(4096) == "0K-4K" and P.ctx_bin(4097) == "4K-8K"
          and P.ctx_bin(200000) == ">128K", "the bin edges are the gates' depths")
    check(m.get("flash-next", {}).get("decode_tps", {}).get("p50") == 15.0,
          "each model its own", json.dumps(m.get("flash-next"))[:200])
    check(sum(b["series"]["n"]) >= 3 and any(v is not None for v in b["series"]["decode_p50"]),
          "a series over the window's buckets", json.dumps(b["series"])[:300])
    # the repository's own bench/results/kv_rank files are read too; this
    # checks the fixture gate
    gates = [g for g in d["gates"]["files"] if g["kind"] == "flash-next gate"]
    check(all("groups" in g for g in d["gates"]["files"]) and not d["gates"]["errors"],
          "every result file found parses", json.dumps(d["gates"]["errors"])[:300])
    check(len(gates) == 1 and gates[0]["source"] == "flashnext-gate-20260929"
          and gates[0]["in_progress"] is True,
          "the gate file is read, and one with no `finished` is in progress",
          json.dumps([{k: g[k] for k in ("source", "in_progress")} for g in gates]))
    grp = {(x["arm"], x["what"]): x for x in gates[0]["groups"]} if gates else {}
    fa = grp.get(("flash-all", "speed/4k"), {})
    check(fa.get("n") == 3 and fa.get("decode", {}).get("median") == 24.0,
          "each arm's runs by depth, with n and the median", json.dumps(fa))
    one = grp.get(("base", "speed_eos_run/32k"), {})
    check(one.get("decode", {}).get("n") == 0 and one.get("prompt", {}).get("median") == 153.0,
          "a run that decoded one token has no decode rate; its prefill still counts",
          json.dumps(one))
    check(gates and gates[0]["pass"].get("arms/base/needles") is True,
          "the gate's PASS flags ride along", json.dumps(gates[0]["pass"] if gates else {}))
    check(isinstance(d["left_out"], list) and d["left_out"], "what was left out, and why")


def main() -> int:
    for fn in (test_off_unless_the_ledger_records, test_hooks_write_rows,
               test_queue_is_bounded, test_jjava_overview,
               test_jjava_injector_and_lane, test_perf_overview):
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
    total = len(_results)
    print(f"\n{'=' * 70}\n  {passed}/{total} checks passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
