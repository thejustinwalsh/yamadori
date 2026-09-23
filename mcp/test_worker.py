#!/usr/bin/env python
"""The worker, asserted. No GPU, no network, no real database, no real corpus.

WHAT THIS IS GATING

  1. A HANDLER THAT RAISES NEVER LANDS IN `done`. Permanent failures go
     straight to `errored`; anything else is retried until attempts run out
     and then `errored`. This is the property jobs.py exists for.
  2. LANES. The gpu lane runs one job at a time, full stop; cpu runs several.
  3. LIVENESS. A running job's heartbeat moves while the handler blocks, and
     a job orphaned by a dead worker is reclaimed at startup and finished.
  4. THE PIPELINE RUNS ITSELF. A URL dataset goes fetch -> extract -> index
     -> complete with no human action after clarify, writes rows with their
     provenance, drops rows whose evidence is not in the source, and builds a
     hints cache whose vectors are not zero (PROTOCOL rule 1).

HOW IT IS ISOLATED

`YAMADORI_JOBS_DB` points at a temp file before `jobs` is imported. The
recipes dir, the datasets dir and the hints corpus/cache are redirected to
temp paths. The model is `worker.ask_model`, replaced; the embedder is a fake
`code_search` module in `sys.modules`, so `hints` never reaches :11434. The
source is served by a local `http.server` on an ephemeral port.
"""
from __future__ import annotations

import http.server
import json
import os
import sys
import tempfile
import threading
import time
import traceback
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

_TMP = tempfile.mkdtemp(prefix="yamadori_test_worker_")
os.environ["YAMADORI_JOBS_DB"] = os.path.join(_TMP, "jobs.sqlite3")
os.environ["YAMADORI_BEAT_SECONDS"] = "0.05"
os.environ["YAMADORI_POLL_SECONDS"] = "0.05"

import numpy as np  # noqa: E402

# The embedder, faked BEFORE hints can import the real one. Deterministic
# unit vectors, unless a test flips ZERO to simulate a dead embedding server.
_fake_cs = types.ModuleType("code_search")
_fake_cs.ZERO = False


def _embed(texts, is_query=False):
    out = []
    for t in texts:
        v = np.zeros(16, dtype=np.float32)
        if not _fake_cs.ZERO:
            v[hash(t) % 16] = 1.0
        out.append(v)
    return np.vstack(out)


_fake_cs.embed = _embed
sys.modules["code_search"] = _fake_cs

import datasets  # noqa: E402
import hints  # noqa: E402
import jobs  # noqa: E402
import worker  # noqa: E402

RECIPES = os.path.join(_TMP, "recipes")
os.makedirs(RECIPES, exist_ok=True)
datasets.RECIPES = RECIPES
hints.CORPUS = RECIPES
hints.CACHE = os.path.join(_TMP, "hints.npz")
worker.DATA_DIR = os.path.join(_TMP, "datasets")

REAL = [os.path.abspath(os.path.join(HERE, "..", "index", n)) for n in
        ("jobs.sqlite3", "hints.npz")]

ANSWERS = {
    "source_name": "Widget Handbook",
    "licence": "CC-BY-4.0",
    "language": "typescript",
    "domains": ["types"],
}

SOURCE_HTML = """<html><head><title>t</title><script>var secret = 1;</script>
<style>p{}</style></head><body>
<h1>Widgets</h1>
<p>Always validate widget props at the component boundary so that a malformed
config fails loudly instead of rendering half a widget.</p>
<p>Prefer composition of small widgets over one configurable mega-widget.</p>
</body></html>"""

VERIFIED = ("Always validate widget props at the component boundary so that "
            "a malformed config fails loudly")

_results: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> bool:
    _results.append((bool(ok), name, detail))
    return bool(ok)


# ---------------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------------
class _Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/missing":
            self.send_response(404)
            self.end_headers()
            return
        body = SOURCE_HTML.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


_server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
threading.Thread(target=_server.serve_forever, daemon=True).start()
BASE = f"http://127.0.0.1:{_server.server_address[1]}"


def model_reply(system, user):
    return json.dumps([
        {"recipe": "Validate widget props at the component boundary.",
         "trigger_condition": "writing a widget that takes config",
         "category": "validation", "confidence": "high",
         "evidence": VERIFIED},
        {"recipe": "Use a global widget registry.",
         "trigger_condition": "always", "category": "made-up",
         "confidence": "high",
         "evidence": "Widgets should always be registered globally at boot"},
    ])


worker.ask_model = model_reply


def register(queue, fn):
    worker.HANDLERS[queue] = fn


def drain() -> int:
    return worker.run(once=True)


# ---------------------------------------------------------------------------
def test_the_fixture_is_isolated():
    check(os.path.abspath(jobs.DB).startswith(os.path.abspath(_TMP)),
          "the jobs database is a temp file", jobs.DB)
    for p in (datasets.RECIPES, hints.CORPUS, hints.CACHE, worker.DATA_DIR):
        check(os.path.abspath(p).startswith(os.path.abspath(_TMP)),
              f"{os.path.basename(p)} is redirected into the temp dir", p)
    check(os.path.abspath(jobs.DB) not in REAL
          and os.path.abspath(hints.CACHE) not in REAL,
          "and neither is the real index")


def test_a_raising_handler_never_lands_in_done():
    def permanent(job, ctx):
        raise worker.Permanent("the thing is missing", "add the thing",
                               owner="operator")
    register("t.permanent", permanent)
    jid = jobs.add("t.permanent", lane="cpu")
    drain()
    j = jobs.get(jid)
    check(j["state"] == "errored",
          "a Permanent failure lands in errored on the first attempt",
          j["state"])
    check(j["attempts"] == 1, "without being retried", str(j["attempts"]))
    check("retryable: no" in (j["error"] or "")
          and "remedy (operator): add the thing" in (j["error"] or ""),
          "and the error carries situation, retryability, remedy and owner",
          str(j["error"]))

    calls = []

    def flaky(job, ctx):
        calls.append(job["attempts"])
        raise OSError("connection reset")
    register("t.flaky", flaky)
    jid = jobs.add("t.flaky", lane="cpu", max_attempts=3)
    drain()
    j = jobs.get(jid)
    check(j["state"] == "errored",
          "an ordinary exception is retried and then errored -- never done",
          j["state"])
    check(calls == [1, 2, 3], "exactly max_attempts times", str(calls))
    check("retryable: yes" in (j["error"] or ""),
          "and it is recorded as retryable, which is the fact", j["error"])
    check(j["result"] is None, "no result was recorded for a failed job",
          str(j["result"]))

    jid = jobs.add("t.no-such-queue", lane="cpu")
    drain()
    j = jobs.get(jid)
    check(j["state"] == "errored" and "no handler" in (j["error"] or ""),
          "a queue with no handler is errored and names the known queues",
          str(j["error"])[:160])


def test_lanes_bound_concurrency():
    lock = threading.Lock()
    live = {"gpu": 0, "cpu": 0}
    peak = {"gpu": 0, "cpu": 0}

    def make(lane):
        def h(job, ctx):
            with lock:
                live[lane] += 1
                peak[lane] = max(peak[lane], live[lane])
            time.sleep(0.25)
            with lock:
                live[lane] -= 1
            return {"ok": True}
        return h
    register("t.gpu", make("gpu"))
    register("t.cpu", make("cpu"))
    ids = [jobs.add("t.gpu", lane="gpu") for _ in range(3)]
    ids += [jobs.add("t.cpu", lane="cpu") for _ in range(8)]
    drain()
    states = {jobs.get(i)["state"] for i in ids}
    check(states == {"done"}, "every lane job finished", str(states))
    check(peak["gpu"] == 1,
          "the gpu lane never ran two jobs at once", str(peak["gpu"]))
    check(1 < peak["cpu"] <= jobs.LANES["cpu"],
          f"the cpu lane ran in parallel, at most {jobs.LANES['cpu']}",
          str(peak["cpu"]))


def test_a_blocking_handler_keeps_its_heartbeat():
    seen = {}

    def slow(job, ctx):
        first = jobs.get(job["id"])["heartbeat"]
        time.sleep(0.4)          # one blocking call, no beat from inside
        seen["moved"] = jobs.get(job["id"])["heartbeat"] > first
        return {}
    register("t.slow", slow)
    jobs.add("t.slow", lane="gpu")
    drain()
    check(seen.get("moved"),
          "the heartbeat advances while the handler is blocked, so a long "
          "generation is not mistaken for a dead worker", str(seen))


def test_an_orphaned_job_is_reclaimed_and_finished():
    register("t.orphan", lambda job, ctx: {"ran": True})
    jid = jobs.add("t.orphan", lane="cpu")
    claimed = jobs.claim("cpu", worker="dead-worker")
    check(claimed and claimed["id"] == jid, "a worker claimed it and died")
    con = jobs._db()
    con.execute("UPDATE jobs SET heartbeat=? WHERE id=?",
                (time.time() - jobs.STALE_SECONDS - 5, jid))
    con.close()
    drain()
    j = jobs.get(jid)
    check(j["state"] == "done" and j["attempts"] == 2,
          "startup reclaim returned it to the queue and it ran, attempt "
          "count intact", json.dumps({k: j[k] for k in ("state", "attempts")}))


def test_html_becomes_text_without_scripts():
    t = worker.html_to_text(SOURCE_HTML)
    check("var secret" not in t and "p{}" not in t,
          "script and style bodies are dropped", t[:120])
    check(worker._norm(VERIFIED) in worker._norm(t),
          "and the prose survives intact enough to verify a quote against")
    parts = worker.chunks_of("a" * 30 + "\n\n" + "b" * 30, size=40)
    check(parts == ["a" * 30, "b" * 30],
          "chunks break on paragraph boundaries", str(parts))


def test_a_url_dataset_runs_end_to_end_unattended():
    ds = datasets.create(f"{BASE}/widgets", name="widget handbook",
                         source_url=f"{BASE}/widgets", **ANSWERS)
    check(ds["stage"] == "clarify", "a submission lands in clarify")
    ran = drain()
    ds = datasets.get(ds["id"])
    check(ds["stage"] == "complete",
          "with every question answered it reaches complete with no human "
          "action: fetch -> extract -> index", f"{ds['stage']} after {ran}")
    rows = jobs.listing(dataset=ds["id"])
    check({j["queue"]: j["state"] for j in rows} == {
              "dataset.fetch": "done", "dataset.extract": "done",
              "dataset.index": "done"},
          "each stage ran exactly once and is done",
          json.dumps([(j["queue"], j["state"]) for j in rows]))

    path = os.path.join(RECIPES, datasets.recipe_file(ds))
    with open(path, encoding="utf-8") as f:
        out = [json.loads(line) for line in f if line.strip()]
    check(len(out) == 1,
          "the recipe whose evidence is in the source is kept; the one whose "
          "quote is not is dropped", str(len(out)))
    if out:
        r = out[0]
        check(r["source_url"] == f"{BASE}/widgets"
              and r["source_name"] == "Widget Handbook"
              and r["license_if_known"] == "CC-BY-4.0",
              "rows carry the dataset's provenance, copied not guessed",
              json.dumps({k: r.get(k) for k in
                          ("source_url", "source_name", "license_if_known")}))
        check(r["_dataset"] == ds["id"] and r["_state"] == "unreviewed",
              "and are marked unreviewed: review is optional, after the fact")
    c = ds["counts"]
    check(c.get("extract_proposed") == 2 and c.get("extract_kept") == 1
          and c.get("extract_dropped_unverified") == 1,
          "the dataset records what extract measured", json.dumps(c))
    check(c.get("indexed_rows") == 1, "and what index embedded",
          json.dumps(c))

    z = np.load(hints.CACHE)
    norms = np.linalg.norm(z["mat"], axis=1)
    check(int(z["n"]) >= 1 and float(norms.min()) > 0.99,
          "the hints cache exists and none of its vectors is zero",
          f"n={int(z['n'])} min={norms.min():.3f}")
    meta = os.path.join(worker.dataset_dir(ds["id"]), "source.json")
    check(os.path.exists(meta), "the fetched source is kept with its hash")


def test_a_rejected_row_leaves_the_hints_corpus():
    path = os.path.join(RECIPES, "reject_probe.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"recipe": "kept row", "_state": "keep"}) + "\n")
        f.write(json.dumps({"recipe": "rejected row",
                            "_state": "reject"}) + "\n")
        f.write(json.dumps({"recipe": "unreviewed row"}) + "\n")
    rows, _ = hints.refresh()
    texts = {r["recipe"] for r in rows if r["_file"] == "reject_probe.jsonl"}
    check(texts == {"kept row", "unreviewed row"},
          "a row marked reject is not served; unreviewed rows are",
          str(sorted(texts)))
    os.remove(path)
    hints.refresh()


def test_zero_vectors_fail_the_index_job():
    _fake_cs.ZERO = True
    try:
        ds = datasets.create("widget advice, pasted", name="zero vec set",
                             source=SOURCE_HTML, source_url="local:paste",
                             **ANSWERS)
        datasets.advance(ds["id"])
        drain()
        idx = [j for j in jobs.listing(dataset=ds["id"])
               if j["queue"] == "dataset.index"]
        check(idx and idx[0]["state"] == "errored",
              "an embedding server returning zeros errors the index job "
              "rather than reporting success (PROTOCOL rule 1)",
              json.dumps([(j["state"], j["error"]) for j in idx])[:240])
        check(datasets.get(ds["id"])["stage"] == "index",
              "and the dataset does not reach complete")
    finally:
        _fake_cs.ZERO = False


def test_extract_refuses_to_overwrite_a_hand_collected_file():
    ds = datasets.create("more widget advice", name="r3f",
                         source="Always validate widget props at the component "
                                "boundary so that a malformed config fails "
                                "loudly.", source_url="local:paste", **ANSWERS)
    path = os.path.join(RECIPES, datasets.recipe_file(ds))
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"recipe": "hand collected"}) + "\n")
    datasets.advance(ds["id"])
    drain()
    ex = [j for j in jobs.listing(dataset=ds["id"])
          if j["queue"] == "dataset.extract"][0]
    check(ex["state"] == "errored" and "did not write" in (ex["error"] or ""),
          "a recipe file this dataset did not write is never overwritten",
          str(ex["error"])[:200])
    with open(path, encoding="utf-8") as f:
        check(f.read().count("hand collected") == 1,
              "and the hand-collected row is untouched")


def test_a_fetch_404_is_permanent_and_blocks_clarify():
    ds = datasets.create(f"{BASE}/missing", name="missing set",
                         source_url=f"{BASE}/missing", **ANSWERS)
    drain()
    f = [j for j in jobs.listing(dataset=ds["id"])][0]
    check(f["state"] == "errored" and f["attempts"] == 1
          and "HTTP 404" in (f["error"] or ""),
          "a 404 is permanent: errored on the first attempt, with the code",
          json.dumps({k: f[k] for k in ("state", "attempts", "error")})[:200])
    check(datasets.get(ds["id"])["stage"] == "clarify",
          "and the dataset stays in clarify rather than extracting nothing")


def test_two_workers_never_run_two_gpu_jobs():
    """Two worker processes overlapped live (a watchdog restart and a manual
    start). The gpu limit must hold across workers, not per process."""
    lock = threading.Lock()
    live = {"n": 0, "peak": 0}

    def gpu(job, ctx):
        with lock:
            live["n"] += 1
            live["peak"] = max(live["peak"], live["n"])
        time.sleep(0.2)
        with lock:
            live["n"] -= 1
        return {}
    register("t.gpu2", gpu)
    ids = [jobs.add("t.gpu2", lane="gpu") for _ in range(4)]
    # Two independent "workers": each runs its own gpu slot thread.
    stop = threading.Event()
    ws = [threading.Thread(target=worker.lane_loop,
                           args=("gpu", k, stop, True)) for k in (7, 8)]
    for t in ws:
        t.start()
    for t in ws:
        t.join()
    drain()
    check(all(jobs.get(i)["state"] == "done" for i in ids),
          "every gpu job still finishes")
    check(live["peak"] == 1,
          "two workers never ran two gpu jobs at once", str(live["peak"]))
    # Directly: with one gpu job running, a second worker's claim is refused.
    jobs.add("t.gpu2", lane="gpu")
    jobs.add("t.gpu2", lane="gpu")
    a = jobs.claim("gpu", worker="worker-a")
    b = jobs.claim("gpu", worker="worker-b")
    check(a is not None and b is None,
          "a second worker's claim on a full gpu lane returns None", str(b))
    if a:
        jobs.finish(a["id"], {})
    drain()


def test_a_dead_local_workers_gpu_slot_is_freed_at_once():
    """With a global lane limit, a killed worker's running gpu row holds the
    only slot. Waiting STALE_SECONDS (20 min) for it is avoidable on the same
    host: its pid is gone. A live pid (this process) must NOT be reclaimed."""
    import socket
    register("t.gpu3", lambda job, ctx: {"ran": True})
    dead_id = jobs.add("t.gpu3", lane="gpu")
    host = socket.gethostname()
    dead = jobs.claim("gpu", worker=f"{host}:999999:gpu0")
    check(dead and dead["id"] == dead_id, "a job is held by a worker pid "
          "that does not exist")
    check(jobs.claim("gpu", worker="other") is None,
          "and it blocks the gpu lane")
    check(not jobs._pid_alive(999999) and jobs._pid_alive(os.getpid()),
          "_pid_alive tells a missing pid from a live one, without killing it")
    n = jobs.reclaim_dead_local()
    check(n == 1 and jobs.get(dead_id)["state"] == "queued",
          "reclaim_dead_local returns it to the queue at once",
          f"{n} {jobs.get(dead_id)['state']}")
    live_id = jobs.add("t.gpu3", lane="gpu")
    drain()
    check(jobs.get(dead_id)["state"] == "done"
          and jobs.get(live_id)["state"] == "done",
          "and the lane drains normally afterwards")
    held = jobs.add("t.gpu3", lane="gpu")
    jobs.claim("gpu", worker=f"{host}:{os.getpid()}:gpu0")
    check(jobs.reclaim_dead_local() == 0
          and jobs.get(held)["state"] == "running",
          "a job held by a LIVE local pid is left alone")
    jobs.finish(held, {})


def test_laya_label_and_train_refuse_with_the_gap_named():
    ds = datasets.create("laya rows", name="laya set", kind="laya",
                         source=SOURCE_HTML, source_url="local:paste",
                         **ANSWERS)
    datasets.advance(ds["id"])
    drain()
    ds = datasets.get(ds["id"])
    label = [j for j in jobs.listing(dataset=ds["id"])
             if j["queue"] == "dataset.label"]
    check(ds["stage"] == "label", "a laya set runs through index to label",
          ds["stage"])
    check(label and label[0]["state"] == "errored"
          and "docs/LAYA.md" in (label[0]["error"] or "")
          and "remedy (developer)" in (label[0]["error"] or ""),
          "label is refused -- errored, never done -- naming the runbook and "
          "a developer as the owner of the gap",
          str(label and label[0]["error"])[:240])


def test_a_paused_lane_hands_out_nothing_and_the_pause_expires():
    """A benchmark pauses the gpu lane so an extraction on the same model does
    not move its seconds and tokens/s. The pause must stop claims, leave other
    lanes alone, and expire on its own if the benchmark dies without resuming."""
    drain()
    jobs.resume("gpu")
    jid = jobs.add("t.pause", lane="gpu")
    cid = jobs.add("t.pause", lane="cpu")
    rec = jobs.pause("gpu", by="test", why="measuring", ttl_seconds=60)
    check(jobs.paused("gpu") and jobs.paused("gpu")["why"] == "measuring",
          "paused() reports the live record", str(jobs.paused("gpu")))
    check(jobs.claim("gpu", worker="w") is None,
          "a paused gpu lane hands out nothing")
    c = jobs.claim("cpu", worker="w")
    check(c is not None and c["id"] == cid, "the cpu lane is untouched", str(c))
    if c:
        jobs.finish(c["id"], {})
    check("gpu" in jobs.snapshot().get("paused", {}),
          "the dashboard snapshot shows the pause")
    jobs.pause("gpu", by="test", why="crashed run", ttl_seconds=-1)
    check(jobs.paused("gpu") is None, "an expired pause is no pause")
    g = jobs.claim("gpu", worker="w")
    check(g is not None and g["id"] == jid,
          "so the lane claims again without anyone resuming it", str(g))
    if g:
        jobs.finish(g["id"], {})
    with open(jobs._pause_path("gpu"), "w", encoding="utf-8") as f:
        f.write("{not json")
    check(jobs.paused("gpu") is None, "a corrupt pause file is no pause")
    check(jobs.resume("gpu") is True and jobs.resume("gpu") is False,
          "resume() removes it once and reports whether there was one")
    check(rec["until"] > rec["since"], "the record carries its expiry")
    drain()


def main() -> int:
    for fn in (test_the_fixture_is_isolated,
               test_a_raising_handler_never_lands_in_done,
               test_lanes_bound_concurrency,
               test_a_blocking_handler_keeps_its_heartbeat,
               test_an_orphaned_job_is_reclaimed_and_finished,
               test_html_becomes_text_without_scripts,
               test_a_url_dataset_runs_end_to_end_unattended,
               test_a_rejected_row_leaves_the_hints_corpus,
               test_zero_vectors_fail_the_index_job,
               test_extract_refuses_to_overwrite_a_hand_collected_file,
               test_a_fetch_404_is_permanent_and_blocks_clarify,
               test_two_workers_never_run_two_gpu_jobs,
               test_a_dead_local_workers_gpu_slot_is_freed_at_once,
               test_laya_label_and_train_refuse_with_the_gap_named,
               test_a_paused_lane_hands_out_nothing_and_the_pause_expires):
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
    print(f"  temp database: {os.path.abspath(jobs.DB)}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
