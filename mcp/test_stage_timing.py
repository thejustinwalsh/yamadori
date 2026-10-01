#!/usr/bin/env python
"""x_yamadori.timing (mcp/stage_timing.py), offline.

  1. A timed call inside a request lands in `stages`; a call it makes lands in `nested` (not summed twice);
     unaccounted = wall - the top-level stages.
  2. The upstream stream: its first event and its total; the server's prompt_ms from the "done" item gives the
     queue estimate; close and exceptions pass through unchanged.
  3. Attribution follows the request's cancel token: a job thread that re-binds it adds to the same timer; a call
     outside any request records nothing and behaves exactly as unwrapped.
  4. The log line appears only past LOG_UNACCOUNTED_S.
  5. install() on the REAL proxy wraps every target that exists, is idempotent, and a wrapped proxy function answers
     as before (the shape the merged proxy lines rely on).
No network, no stack: the fake module below stands in for the upstream.
"""
from __future__ import annotations

import os
import sys
import threading
import time
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import offline_stores  # noqa: E402

offline_stores.isolate("yamadori_test_stage_timing_")
os.environ.setdefault("YAMADORI_GPU_ROOM", "0")

import cancel  # noqa: E402
import stage_timing as st  # noqa: E402

_results: list[tuple[bool, str, str]] = []


def check(ok, name, detail=""):
    _results.append((bool(ok), name, str(detail)))


def fake_module():
    m = types.ModuleType("fakeproxy")

    def inner(x):
        time.sleep(0.05)
        return x * 2

    def outer(x):
        time.sleep(0.05)
        return m.inner(x) + 1

    def stream(n, fail=False):
        time.sleep(0.1)                        # "queued" before the first event
        for i in range(n):
            yield ("delta", i)
        if fail:
            raise ValueError("upstream broke")
        yield ("done", {"_timings": {"prompt_ms": 20.0}})
    m.inner, m.outer, m.stream = inner, outer, stream
    return m


def test_stages_nested_and_unaccounted():
    m = fake_module()
    st.install([("fakeproxy", "outer", "call"), ("fakeproxy", "inner", "call")], modules={"fakeproxy": m})
    tok = cancel.Token()
    with cancel.bound(tok):
        st.begin()
        r = m.outer(3)
        time.sleep(0.12)                       # unaccounted: nothing timed runs here
        rec = st.record()
    check(r == 7, "a wrapped call answers exactly as before", r)
    check(set(rec["stages"]) == {"fakeproxy.outer"} and "fakeproxy.inner" in rec["nested"],
          "the outer call is a stage; the call it makes is nested detail, not summed twice", rec)
    check(95 <= rec["stages"]["fakeproxy.outer"] < 400 and rec["unaccounted_ms"] >= 100,
          "unaccounted = wall - the top-level stages (the 120 ms sleep)", rec)
    with cancel.bound(cancel.Token()):
        check(m.outer(1) == 3 and st.record() == {}, "a request that never began records nothing")
    check(m.outer(2) == 5, "outside any request: the plain call")


def test_stream():
    m = fake_module()
    st.install([("fakeproxy", "stream", "gen")], modules={"fakeproxy": m})
    tok = cancel.Token()
    with cancel.bound(tok):
        st.begin()
        got = list(m.stream(3))
        rec = st.record()
    p = rec["points"]
    check(got[-1][0] == "done" and len(got) == 4, "every event passes through", got)
    check(p.get("upstream_first_event_ms", 0) >= 95 and p.get("upstream_prompt_ms") == 20.0
          and p.get("upstream_queue_est_ms", 0) >= 75,
          "first event, the server's prompt_ms and the queue estimate (first event less prompt_ms)", p)
    check("upstream.generate" in rec["stages"], "the stream is a stage", rec["stages"])
    with cancel.bound(cancel.Token()):
        st.begin()
        g = m.stream(5)
        next(g)
        g.close()
        rec2 = st.record()
    check("upstream.generate" in rec2["stages"], "a stream closed early (a hang-up) still records its time")
    with cancel.bound(cancel.Token()):
        st.begin()
        try:
            list(m.stream(2, fail=True))
            check(False, "an upstream error passes through")
        except ValueError:
            check(True, "an upstream error passes through unchanged")


def test_job_threads_share_the_timer():
    m = fake_module()
    st.install([("fakeproxy", "inner", "call")], modules={"fakeproxy": m})
    tok = cancel.Token()
    with cancel.bound(tok):
        st.begin()

        def job():
            with cancel.bound(tok):             # how the second brain's threads re-bind
                m.inner(1)
        ths = [threading.Thread(target=job) for _ in range(3)]
        [t.start() for t in ths]
        [t.join() for t in ths]
        rec = st.record()
    check(rec["calls"].get("fakeproxy.inner") == 3, "a job thread bound to the request adds to its timer", rec)


def test_log_line():
    quiet = {"wall_ms": 1000.0, "unaccounted_ms": 100.0, "stages": {}}
    loud = {"wall_ms": 72800.0, "unaccounted_ms": 70000.0, "stages": {"upstream.generate": 800.0},
            "points": {"upstream_first_event_ms": 600.0, "upstream_prompt_ms": 366.0}}
    check(st.log_if_slow(quiet) is None, "no line below LOG_UNACCOUNTED_S")
    line = st.log_if_slow(loud, "side call")
    check(line and "70.0s unaccounted" in line and "upstream.generate" in line,
          "one line past it, with the largest stages and the upstream first event", line)


def test_install_on_the_real_proxy():
    import proxy
    before = proxy.resolve_repo
    out = st.install()
    again = st.install()
    check(out.get("proxy.prepare") == "wrapped" and out.get("proxy._post_events_raw") == "wrapped"
          and out.get("slots.acquire") == "wrapped",
          "install wraps the turn's stages on the real modules", {k: v for k, v in out.items() if v != "wrapped"})
    check(all(v in ("already", "missing") for v in again.values()), "install is idempotent")
    check(proxy.resolve_repo is not before and proxy.resolve_repo.__wrapped__ is before,
          "the wrapper keeps the original (functools.wraps)")
    msgs = [{"role": "user", "content": "hi"}]
    with cancel.bound(cancel.Token()):
        st.begin()
        r1 = proxy.resolve_repo(msgs, "")
        rec = st.record()
    check(r1 == before(msgs, "") and "proxy.resolve_repo" in rec["stages"],
          "a wrapped proxy function answers as before and is timed", rec["stages"])


def main() -> int:
    for fn in (test_stages_nested_and_unaccounted, test_stream, test_job_threads_share_the_timer, test_log_line,
               test_install_on_the_real_proxy):
        try:
            fn()
        except Exception as e:                                       # noqa: BLE001
            import traceback
            check(False, f"{fn.__name__} raised", f"{type(e).__name__}: {e}\n{traceback.format_exc()[-1200:]}")
    bad = [r for r in _results if not r[0]]
    for ok, name, detail in _results:
        print(("ok    " if ok else "FAIL  ") + name + (f"  ({detail[:400]})" if not ok and detail else ""))
    print(f"\n{'=' * 70}\n  {len(_results) - len(bad)}/{len(_results)} checks passed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
