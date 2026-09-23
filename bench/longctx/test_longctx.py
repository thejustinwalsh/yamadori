#!/usr/bin/env python
"""Offline proofs for bench/longctx: the haystack, the grader, the runner's
resume and error handling, and the usable-context analysis.

    python bench/longctx/test_longctx.py

No GPU and no live service: the runner is driven against an in-process fake
llama-server on a loopback port that answers from the haystack it is sent.
Prints "N/M checks passed" (scripts/run_tests.py reads it).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import analyse  # noqa: E402
import haystack as hs  # noqa: E402
import run as runner  # noqa: E402

PASSED, TOTAL = 0, 0


def check(cond, what: str) -> None:
    global PASSED, TOTAL
    TOTAL += 1
    if cond:
        PASSED += 1
    else:
        print(f"  FAIL: {what}")


def synthetic_chunks(n: int = 300) -> list[tuple[str, str]]:
    out = []
    for i in range(n):
        body = "\n".join(
            f"export function helper{i}_{j}(a, b) {{\n  const v = a * {j} + b;\n"
            f"  return v > {i} ? v : {i};\n}}\n" for j in range(12))
        out.append((f"pkg/file{i}.js", body))
    return out


# ------------------------------------------------------------- haystack ----
def test_haystack():
    ch = synthetic_chunks()
    it1 = hs.make_item(7, 3)
    it2 = hs.make_item(7, 3)
    check(it1 == it2, "make_item is deterministic")
    check(hs.make_item(7, 4) != it1, "a different item differs")
    check(hs.make_item(8, 3) != it1, "a different seed differs")
    a = hs.build_haystack(it1, 60_000, ch)
    b = hs.build_haystack(it2, 60_000, ch)
    check(a["text"] == b["text"], "build_haystack is byte-deterministic")
    check(abs(a["chars"] - 60_000) < 2_000, f"haystack size near target ({a['chars']})")
    names = [n["name"] for n in it1["needles"]] + [d["name"] for d in it1["decoys"]]
    check(all(a["text"].count(f"export function {n}()") == 1 for n in names),
          "every needle and decoy is inserted exactly once")
    check(len(set(names)) == 8 and all(x not in y for x in names for y in names if x != y),
          "no function name is a substring of another")
    vals = [n["value"] for n in it1["needles"]] + [d["value"] for d in it1["decoys"]]
    check(len(set(vals)) == len(vals), "all inserted values distinct")
    check(sorted(n["depth"] for n in it1["needles"]) ==
          sorted(set(n["depth"] for n in it1["needles"])) and len(it1["needles"]) == 4,
          "four needles at four distinct depth bins")
    check(it1["single"]["depth"] == hs.DEPTH_BINS[3 % 5],
          "single question targets bin (item mod 5)")
    # Pairing: the item is the same at every length.
    small = hs.build_haystack(it1, 20_000, ch)
    check(all(f"return {n['value']};" in small["text"] for n in it1["needles"]),
          "the same needles appear at a different length")
    # Depths, by characters and by a crude token count.
    worst_c, worst_t = 0.0, 0.0
    for size in (20_000, 120_000, 400_000):
        h = hs.build_haystack(hs.make_item(3, 11), size, ch)
        toks = [(m.start()) for m in re.finditer(r"\w+|[^\w\s]", h["text"])]
        for nm, v in h["landed"].items():
            if v["kind"] != "needle":
                continue
            worst_c = max(worst_c, abs(v["actual"] - v["target"]))
            pos = h["text"].index(f"export function {nm}()")
            before = sum(1 for p in toks if p < pos)
            worst_t = max(worst_t, abs(before / len(toks) - v["target"]))
    check(worst_c <= 0.02, f"needle char depth within 2 points of target (worst {worst_c:.4f})")
    check(worst_t <= 0.03, f"needle token-proxy depth within 3 points (worst {worst_t:.4f})")
    # Insertion never splits a line.
    h = hs.build_haystack(it1, 80_000, ch)
    check(all(h["text"][h["text"].index(f"export function {n}()") - 1] == "\n"
              for n in names), "every insert starts at a line boundary")


def test_real_sources():
    ch = hs.load_chunks()
    if not ch:
        print("  note: index/packages/_src absent; real-source checks skipped")
        return
    check(hs.corpus_chars(ch) > 1_200_000,
          f"real sources cover a 256k-token haystack ({hs.corpus_chars(ch):,} chars)")
    it = hs.make_item(1, 0)
    h = hs.build_haystack(it, 300_000, ch)
    worst = max(abs(v["actual"] - v["target"]) for v in h["landed"].values()
                if v["kind"] == "needle")
    check(worst <= 0.01, f"real-source needle depth within 1 point (worst {worst:.4f})")
    check(h["text"] == hs.build_haystack(it, 300_000, hs.load_chunks())["text"],
          "real-source haystack is deterministic across loads")
    check("// ==== file: " in h["text"], "files are labelled")


# ------------------------------------------------------------- grader ------
def test_grader():
    it = hs.make_item(5, 2)
    s = it["single"]
    g = hs.grade(it, "single", f"It returns {s['value']}.\nANSWER: {s['value']}", "stop")
    check(g["outcome"] == "correct" and g["parse"] == "answer_line", "single correct")
    g = hs.grade(it, "single", f"ANSWER: {s['value'] + 1}", "stop")
    check(g["outcome"] == "wrong", "single off-by-one is wrong")
    decoy = next(d for d in it["decoys"] if d["of"] == s["name"])
    g = hs.grade(it, "single", f"ANSWER: {decoy['value']}", "stop")
    check(g["outcome"] == "wrong", "the decoy's value is wrong")
    g = hs.grade(it, "single", f"ANSWER: `{s['value']:,}`", "stop")
    check(g["outcome"] == "correct", "thousands separator and backticks tolerated")
    g = hs.grade(it, "single", f"the value is {s['value']}", "stop")
    check(g["outcome"] == "correct" and g["parse"] == "fallback",
          "no ANSWER line: last integer, marked fallback")
    g = hs.grade(it, "single", "", "length")
    check(g["outcome"] == "budget" and g["correct"] is None,
          "length with no answer is a budget event, not scored")
    g = hs.grade(it, "single", "", "stop")
    check(g["outcome"] == "wrong", "empty reply with stop is wrong")
    q = it["reason"]
    g = hs.grade(it, "reason", f"ANSWER: {q['value']}", "stop")
    check(g["outcome"] == "correct", "reason correct")
    g = hs.grade(it, "reason", f"ANSWER: 1 + 2 = {q['value']}", "stop")
    check(g["outcome"] == "correct", "reason: last integer on the ANSWER line")
    g = hs.grade(it, "reason", f"ANSWER: {q['value'] + 10}", "stop")
    check(g["outcome"] == "wrong", "reason wrong")
    exp = hs.expected(it, "multi")
    line = ", ".join(f"{n}={v}" for n, v in exp.items())
    g = hs.grade(it, "multi", "thinking...\nANSWER: " + line, "stop")
    check(g["outcome"] == "correct" and all(g["per_needle"].values()), "multi correct")
    styled = ", ".join(f"`{n}()`: **{v}**" for n, v in exp.items())
    g = hs.grade(it, "multi", "ANSWER: " + styled, "stop")
    check(g["outcome"] == "correct", "multi tolerates backticks, () and bold")
    first = next(iter(exp))
    bad = ", ".join(f"{n}={v + (1 if n == first else 0)}" for n, v in exp.items())
    g = hs.grade(it, "multi", "ANSWER: " + bad, "stop")
    check(g["outcome"] == "wrong" and sum(g["per_needle"].values()) == 3,
          "multi with one wrong value is wrong, 3/4 per needle")
    part = ", ".join(f"{n}={v}" for n, v in list(exp.items())[:3])
    g = hs.grade(it, "multi", "ANSWER: " + part, "stop")
    check(g["outcome"] == "wrong", "multi missing one name is wrong")
    dec = {d["of"]: d for d in it["decoys"]}
    swapped = ", ".join(f"{dec[n]['name']}={v}" for n, v in exp.items())
    g = hs.grade(it, "multi", "ANSWER: " + swapped, "stop")
    check(g["outcome"] == "wrong", "decoy names do not satisfy the needles")


# ------------------------------------------------------------- fake server -
class Fake:
    """A llama-server stand-in. Answers from the haystack it receives."""

    def __init__(self, n_ctx=3000, fail_calls=(), always_fail=False,
                 overflow_at=None, speculate=None):
        self.n_ctx = n_ctx
        self.fail_calls = set(fail_calls)
        self.always_fail = always_fail
        self.overflow_at = overflow_at
        # None: no speculation. "timings": draft_n in the reply's timings.
        # "metrics": only the Prometheus counters on /metrics move.
        self.speculate = speculate
        self.drafted, self.accepted = 0, 0
        self.slot = ""                        # the one slot's cached prompt
        self.calls = 0
        self.bodies = []
        fake = self

        def accept_for(q: str) -> tuple[int, int]:
            # Content-dependent acceptance, so the by-content table has
            # something to separate.
            if "JSON" in q:
                return 10, 9
            if "TypeScript module" in q:
                return 10, 6
            if "Summarise" in q:
                return 10, 7
            return 10, 8

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, obj):
                data = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path == "/props":
                    self._send(200, {"default_generation_settings": {"n_ctx": fake.n_ctx},
                                     "model_path": "C:/m/fake-model.gguf",
                                     "build_info": "b1-fake", "total_slots": 1})
                elif self.path == "/metrics" and fake.speculate == "metrics":
                    data = ("# HELP x\n"
                            f"llamacpp:spec_decode_num_draft_tokens_total {fake.drafted}\n"
                            f"llamacpp:spec_decode_num_accepted_tokens_total {fake.accepted}\n"
                            "llamacpp:prompt_tokens_total 1\n").encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                else:
                    self._send(404, {"error": "nope"})

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n))
                fake.calls += 1
                fake.bodies.append(body)
                text = "".join(m["content"] for m in body["messages"])
                pt = len(text) // 3 + 20
                if fake.always_fail or fake.calls in fake.fail_calls:
                    return self._send(500, {"error": {"message": "boom"}})
                if fake.overflow_at and pt > fake.overflow_at:
                    return self._send(400, {"error": {"message":
                                     "the request exceeds the available context size"}})
                user = body["messages"][-1]["content"]
                found = dict(re.findall(r"export function (\w+)\(\) \{\n  return (\d+);",
                                        user))
                q = user.rsplit("</code>", 1)[-1]
                names = re.findall(r"`(\w+)`", q)
                if "For each one" in q:
                    ans = "ANSWER: " + ", ".join(f"{x}={found.get(x, 0)}" for x in names)
                elif "Let x be" in q:
                    a, b = int(found.get(names[0], 0)), int(found.get(names[1], 0))
                    ans = f"ANSWER: {a + b if 'x + y' in q else abs(a - b)}"
                elif names:
                    ans = f"ANSWER: {found.get(names[0], 0)}"
                else:
                    ans = "The code defines helpers. " * 10
                mt = body.get("max_tokens") or 64
                pred = mt if body.get("ignore_eos") else 12
                common = 0
                if body.get("cache_prompt"):
                    while (common < min(len(text), len(fake.slot))
                           and text[common] == fake.slot[common]):
                        common += 1
                fake.slot = text
                cache_n = common // 3
                timings = {"prompt_n": pt - cache_n, "cache_n": cache_n,
                           "prompt_ms": pt / 2, "prompt_per_second": 2000.0,
                           "predicted_n": pred, "predicted_ms": pred * 20,
                           "predicted_per_second": 50.0}
                if fake.speculate:
                    dn, da = accept_for(q)
                    fake.drafted += dn
                    fake.accepted += da
                    if fake.speculate == "timings":
                        timings.update(draft_n=dn, draft_n_accepted=da)
                self._send(200, {
                    "choices": [{"index": 0, "finish_reason": "stop",
                                 "message": {"role": "assistant", "content": ans}}],
                    "usage": {"prompt_tokens": pt, "completion_tokens": pred},
                    "timings": timings})

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def close(self):
        self.srv.shutdown()
        self.srv.server_close()


def make_cfg(fake: Fake, client, **over) -> dict:
    class A:
        pass
    a = A()
    a.mode, a.features, a.model, a.max_ctx = "server", None, "m", None
    a.thinking, a.decode_tokens = "off", 32
    a.ladder, a.items, a.speed_reps, a.seed = "1k,2k,4k", 3, 2, 1
    a.tasks, a.sources, a.gpu_index = "single,multi,reason", None, 0
    a.label, a.run_id, a.speed_content = "T", "t", None
    for k, v in over.items():
        setattr(a, k, v)
    cfg = runner.build_cfg(a, client, client.props())
    cfg["cal"] = runner.calibrate(client, cfg, synthetic_chunks())
    return cfg


def quiet(*_a, **_k):
    pass


def test_runner():
    runner.vram_used = lambda idx: None      # no nvidia-smi in a unit test
    ch = synthetic_chunks()
    tmp = tempfile.mkdtemp(prefix="longctx_test_")
    try:
        # 1. a clean run: every rung that fits, none that does not.
        fake = Fake(n_ctx=3000)
        cl = runner.Client(fake.url, None, None, timeout=30)
        check(cl.props() is not None, "fake /props answers like llama-server")
        cfg = make_cfg(fake, cl)
        check(abs(cfg["cal"]["chars_per_token"] - 3.0) < 0.05,
              f"calibration recovers chars/token ({cfg['cal']['chars_per_token']:.3f})")
        check(cfg["limit"] == 3000, "limit is the server's n_ctx")
        d1 = os.path.join(tmp, "a")
        res = runner.run(cl, cfg, d1, ch, log=quiet)
        rows, bad = runner.load_rows(os.path.join(d1, "rows.jsonl"))
        Ls = sorted({r["L"] for r in rows})
        check(Ls == [1024, 2048], f"ladder stops cleanly at n_ctx (ran {Ls})")
        check(res["ran"] == 2 * (2 * 3 + 3 * 3) and bad == 0,
              f"one row per (L, task, item) ({res['ran']})")
        acc = [r for r in rows if not r["task"].startswith("speed")]
        check(all(r["outcome"] == "correct" for r in acc),
              "a server that reads the haystack scores 100% "
              f"({sum(r['outcome'] == 'correct' for r in acc)}/{len(acc)})")
        spd = [r for r in rows if r["task"].startswith("speed")]
        check(sorted({r["content_type"] for r in spd}) == ["code", "json", "prose"],
              "each speed rep decodes prose, code and json")
        check(all(r["speed_source"] == "server_timings" and r["decode_tps"] == 50.0
                  and r["prefill_tps"] == 2000.0 for r in spd),
              "speed rows take llama-server's timings")
        cold = [r for r in spd if r["cold"]]
        warm = [r for r in spd if not r["cold"]]
        check(len(cold) == 4 and all(r["content_type"] == "prose" for r in cold)
              and all(r["cache_n"] <= 5 for r in cold),
              "one cold (fresh-nonce) prefill per rep, and the server reused nothing "
              f"({[r['cache_n'] for r in cold]})")
        check(len(warm) == 8 and all(r["cache_n"] > 0.8 * r["prompt_tokens"] for r in warm),
              "the rep's other content types reuse the cached haystack")
        check(all(analyse.is_cold(r) for r in cold) and not any(analyse.is_cold(r) for r in warm),
              "analysis takes prefill only from cold rows")
        sb = [b for b in fake.bodies if b.get("ignore_eos")]
        check(len(sb) == 12 and all(b["max_tokens"] == 32 for b in sb),
              "speed requests: ignore_eos and a fixed decode length")
        heads = {b["messages"][0]["content"] for b in sb}
        check(len(heads) == 4, "a unique nonce per speed rep, shared by its content types")
        check(all(r["draft_accept_rate"] is None and r["draft_source"] is None for r in rows),
              "no speculation: draft acceptance is null on every row")
        ab = [b for b in fake.bodies if not b.get("ignore_eos")][2:]
        check(all(b["chat_template_kwargs"] == {"enable_thinking": False}
                  and b["temperature"] == 0 for b in ab),
              "thinking off and temperature 0 are sent explicitly")
        check(all(r["thinking"] == "off" and r["model"] == "fake-model.gguf"
                  and r["build"] == "b1-fake" for r in rows),
              "model id, build and thinking recorded per row")
        check(all(abs(r["prompt_tokens"] / r["L"] - 1) < 0.08 for r in acc),
              "actual prompt tokens land near the target L")
        calls = fake.calls
        res2 = runner.run(cl, cfg, d1, ch, log=quiet)
        check(res2["ran"] == 0 and fake.calls == calls, "a rerun of a finished run sends nothing")
        fake.close()

        # 2. a stack error is recorded, never scored, and retried on resume.
        # 2 calibration calls, 6 speed calls, then item 0 single, MULTI.
        fake = Fake(n_ctx=3000, fail_calls={2 + 6 + 2})
        cl = runner.Client(fake.url, None, None, timeout=30)
        cfg = make_cfg(fake, cl, ladder="1k")
        d2 = os.path.join(tmp, "b")
        runner.run(cl, cfg, d2, ch, log=quiet)
        rows, _ = runner.load_rows(os.path.join(d2, "rows.jsonl"))
        errs = [r for r in rows if r["outcome"] == "stack_error"]
        check(len(errs) == 1 and errs[0]["stack_error_kind"] == "http_500",
              "a 500 becomes one stack_error row: "
              + "; ".join(f"{r['task']}#{r['item']} {r.get('stack_error_kind')} "
                          f"{str(r.get('error'))[:80]}" for r in errs))
        check("correct" not in errs[0] or errs[0].get("correct") is None,
              "a stack_error row carries no score")
        done = runner.done_keys(rows)
        check(errs[0]["key"] not in done and len(done) == len(rows) - 1,
              "done_keys excludes the errored key")
        last = analyse.load_run(d2)["last"]
        cell = analyse.acc_cell(last, 1024, errs[0]["task"])
        check(cell["stack_errors"] == 1 and cell["n"] == 2 and cell["k"] == 2,
              f"analysis excludes the stack error from n ({cell})")
        todo = runner.plan(cfg, done)
        check(todo == [(1024, errs[0]["task"], errs[0]["item"])],
              f"resume plans exactly the errored key ({todo})")
        runner.run(cl, cfg, d2, ch, log=quiet)
        last = analyse.load_run(d2)["last"]
        cell = analyse.acc_cell(last, 1024, errs[0]["task"])
        check(cell["stack_errors"] == 0 and cell["n"] == 3,
              "after the retry the key is scored")
        fake.close()

        # 3. an outage stops the run instead of filling it with errors.
        fake = Fake(n_ctx=3000)
        cl = runner.Client(fake.url, None, None, timeout=30)
        cfg = make_cfg(fake, cl, ladder="1k")
        fake.always_fail = True
        d3 = os.path.join(tmp, "c")
        res = runner.run(cl, cfg, d3, ch, log=quiet)
        rows, _ = runner.load_rows(os.path.join(d3, "rows.jsonl"))
        check(res["stopped"] and "outage" in res["stopped"]
              and len(rows) == runner.MAX_CONSECUTIVE_ERRORS,
              "consecutive stack errors abort the run")
        fake.close()

        # 4. a context overflow ends the ladder at that rung.
        fake = Fake(n_ctx=100_000)
        cl = runner.Client(fake.url, None, None, timeout=30)
        cfg = make_cfg(fake, cl, ladder="1k,2k,4k", items=1, speed_reps=1)
        fake.overflow_at = 1500
        d4 = os.path.join(tmp, "d")
        res = runner.run(cl, cfg, d4, ch, log=quiet)
        rows, _ = runner.load_rows(os.path.join(d4, "rows.jsonl"))
        check(res["blocked_L"] == 2048 and max(r["L"] for r in rows) == 2048
              and any(r.get("stack_error_kind") == "context_exceeded" for r in rows),
              "context_exceeded blocks every longer rung")
        fake.close()

        # 5. speculation: acceptance per row, from timings or from /metrics.
        for mode in ("timings", "metrics"):
            fake = Fake(n_ctx=3000, speculate=mode)
            cl = runner.Client(fake.url, None, None, timeout=30)
            cfg = make_cfg(fake, cl, ladder="1k,2k", items=2, speed_reps=1)
            check(cfg["metrics"] == (mode == "metrics"),
                  f"{mode}: /metrics detected only when exposed")
            d5 = os.path.join(tmp, f"spec_{mode}")
            runner.run(cl, cfg, d5, ch, log=quiet)
            rows, _ = runner.load_rows(os.path.join(d5, "rows.jsonl"))
            want = "timings" if mode == "timings" else "metrics_delta"
            check(all(r["draft_source"] == want and r["draft_n"] == 10 for r in rows),
                  f"{mode}: every row carries its own draft count ({want})")
            byc = {r["content_type"]: r["draft_accept_rate"] for r in rows}
            check(byc.get("json") == 0.9 and byc.get("code") == 0.6
                  and byc.get("prose") == 0.7 and byc.get("answer:multi") == 0.8,
                  f"{mode}: acceptance is per request, not a running total ({byc})")
            dt = analyse.draft_table(analyse.load_run(d5)["last"])
            cell = dt["table"][1]["by_content"]
            check(dt["speculating"] and cell["json"]["rate"] == 0.9
                  and cell["code"]["rate"] == 0.6 and [r["L"] for r in dt["table"]] == [1024, 2048],
                  f"{mode}: analysis tabulates acceptance by content and length")
            fake.close()
        dt = analyse.draft_table(analyse.load_run(d1)["last"])
        check(not dt["speculating"] and all(c["rate"] is None for row in dt["table"]
                                            for c in row["by_content"].values()),
              "analysis leaves acceptance null when nothing was drafted")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_helpers():
    check(runner.parse_ladder("2k,8k, 16k,2k,1000") == [1000, 2048, 8192, 16384],
          "ladder parsing")
    check(runner.row_key(8192, "multi", 3, 1) == "8192|multi|3|1", "row key shape")
    t = {"prompt_per_second": 900.0, "predicted_per_second": 40.0, "prompt_n": 100,
         "predicted_n": 256, "draft_n": 200, "draft_n_accepted": 150}
    s = runner.speed_of({"timings": t})
    check(s["speed_source"] == "server_timings" and s["draft_accept_rate"] == 0.75,
          "draft acceptance from timings")
    m = runner.parse_draft_metrics(
        "# HELP llamacpp:spec_decode_num_draft_tokens_total x\n"
        "llamacpp:spec_decode_num_draft_tokens_total 40\n"
        "llamacpp:spec_decode_num_accepted_tokens_total 30\n")
    check(m == (40.0, 30.0), "Prometheus draft counters parse")
    check(runner.parse_draft_metrics("llamacpp:prompt_tokens_total 5\n") is None,
          "no spec_decode counters: None, not zero")
    d = runner.draft_of(None, (40.0, 30.0), (50.0, 37.0))
    check(d["draft_source"] == "metrics_delta" and d["draft_accept_rate"] == 0.7,
          "acceptance from the /metrics delta")
    d = runner.draft_of({"predicted_n": 5}, (40.0, 30.0), (40.0, 30.0))
    check(d["draft_accept_rate"] is None and d["draft_source"] is None,
          "nothing drafted: null, never 0% acceptance")
    s = runner.speed_of({"usage": {"prompt_tokens": 1000, "completion_tokens": 101},
                         "t0": 0.0, "t_first": 2.0, "t_last": 4.0})
    check(s["speed_source"] == "wallclock_stream" and s["prefill_tps"] == 500.0
          and s["decode_tps"] == 50.0, "wall-clock fallback is labelled and derived")


# ------------------------------------------------------------- analysis ----
def synth_rows(acc_by_L: dict[int, int], n: int = 20, task: str = "single") -> dict:
    last = {}
    for L, k in acc_by_L.items():
        for i in range(n):
            ok = i < k
            r = {"key": f"{L}|{task}|{i}|1", "L": L, "task": task, "item": i, "seed": 1,
                 "outcome": "correct" if ok else "wrong", "prompt_tokens": L}
            last[r["key"]] = r
    return last


def test_analysis():
    check(analyse.min_discordant(0.05) == 6, "min discordant at alpha 0.05 is 6")
    check(analyse.min_discordant(0.05 / 5) == 8, "min discordant at alpha 0.01 is 8")
    last = synth_rows({8192: 19, 16384: 19, 32768: 18, 65536: 9, 98304: 5})
    u = analyse.usable_context(last, tasks=("single",))
    check(u["ref_L"] == 8192 and u["usable"] == 32768,
          f"usable context stops before the significant drop ({u['usable']})")
    t64 = next(t for t in u["by_task"]["single"]["tests"] if t["L"] == 65536)
    check(t64["significant_drop"] and t64["ref_right_L_wrong"] == 10,
          "the 64k drop is significant and counted pairwise")
    last = synth_rows({8192: 19, 16384: 19, 32768: 19})
    u = analyse.usable_context(last, tasks=("single",))
    check(u["usable"] == 32768, "no drop: usable is the longest rung")
    # A drop that the corrected test cannot see is a warning, not a verdict.
    last = synth_rows({8192: 20, 16384: 20, 32768: 20, 65536: 13, 98304: 20, 131072: 20})
    u = analyse.usable_context(last, tasks=("single",))
    check(u["usable"] == 131072 and any(w["L"] == 65536 for w in u["early_warnings"]),
          "an uncorrected-only drop is an early warning")
    # Improvement at a longer rung is never a drop.
    last = synth_rows({8192: 10, 16384: 20})
    u = analyse.usable_context(last, tasks=("single",))
    check(u["usable"] == 16384, "a longer rung doing better is not a drop")
    pw = analyse.power_note(20, 5, 0.9)
    check(pw["min_detectable_drop_corrected"] == 0.4
          and pw["min_detectable_drop_uncorrected"] == 0.3,
          f"power at n=20, m=5: 40 points corrected, 30 uncorrected ({pw})")
    # Speed table: sources are not mixed silently.
    last = {"a": {"key": "a", "L": 2048, "task": "speed", "outcome": "measured",
                  "prefill_tps": 1000.0, "decode_tps": 50.0, "speed_source": "server_timings",
                  "prompt_tokens": 2050},
            "b": {"key": "b", "L": 2048, "task": "speed", "outcome": "stack_error"}}
    st = analyse.speed_table(last)
    check(st[0]["n"] == 1 and st[0]["errors"] == 1 and st[0]["prefill_tps"] == 1000.0,
          "speed table counts errors apart from measurements")


def main() -> int:
    for t in (test_haystack, test_real_sources, test_grader, test_helpers,
              test_runner, test_analysis):
        try:
            t()
        except Exception as e:                                   # noqa: BLE001
            import traceback
            traceback.print_exc()
            check(False, f"{t.__name__} raised {type(e).__name__}: {e}")
    print(f"{PASSED}/{TOTAL} checks passed")
    return 0 if PASSED == TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
