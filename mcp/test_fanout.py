#!/usr/bin/env python
"""Fan-out: the sequential second brain, its grade, and the vote. No GPU.

WHAT THIS IS GATING

  1. SEQUENTIAL, ONE SECOND BRAIN (operator decision, 2026-09-23). The
     original answer is A; the helper writes B under admission.helper_lane()
     with the helper's whole 3/8 (share_n=1); the two are GRADED:
       - exactly one parses              -> it wins, 2 steps
       - both parse, similarity >= AGREE -> A wins, 2 steps
       - otherwise                       -> the tie-breaker C, whose prompt
         carries the task, both candidates' code and each one's check result
     At most one helper context is ever live, even across requests.
  2. EVERY second-brain run carries a fresh concept seed in its user turn
     (operator, 2026-09-23): B, C and deep thinking. Each is recorded, per
     candidate in x_yamadori.fanout.candidates[].seed and in the deep-thinking
     record, and concept_seed.record() updates the dashboard's last seed.
  3. A prose answer keeps the original: B runs, the vote is recorded only.
  4. A helper lane that never comes free is recorded, skipped: "helper busy".
  5. NO VOTES IS NOT DISAGREEMENT (docs/SELECTION-BUILD.md harm 2), and
     observed disagreement is still reported.
  6. A CODE answer is not picked by length: the medoid of the parsing
     candidates (fanout.consensus), the tie-breaker preferred on a tie;
     nothing parsing falls back, flagged; candidate code is never executed.

The model is a fake upstream (a real http.server speaking SSE, so the
proxy's own reader runs); consensus() is tested on fabricated candidates.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

_TMP = tempfile.mkdtemp(prefix="yamadori_test_fanout_")
os.environ["YAMADORI_CORPUS_DB"] = os.path.join(_TMP, "corpus.sqlite3")
os.environ["YAMADORI_NEBARI_DB"] = os.path.join(_TMP, "nebari.sqlite3")
os.environ["RINGS_DB"] = os.path.join(_TMP, "rings.sqlite3")
os.environ["YAMADORI_PKG_DIR"] = os.path.join(_TMP, "pkgs")
os.environ["CODE_INDEX_DB"] = os.path.join(_TMP, "code.sqlite3")
os.environ["CONCEPT_SEED_LAST"] = os.path.join(_TMP, "seed_last.json")
# Nothing here may reach a model. A reserved port refuses at once.
os.environ["LLAMA_STACK_URL"] = "http://127.0.0.1:1"

import admission  # noqa: E402
import budget  # noqa: E402
import concept_seed  # noqa: E402
import fanout  # noqa: E402
import proxy  # noqa: E402
import shomen  # noqa: E402
import tiers  # noqa: E402

# budget.pool_size() would ask the live server. Pinned to the shipped `-c`.
budget._POOL = 147456
tiers._accepted = tiers.FALLBACK_EFFORTS     # no network for the template

_results: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> bool:
    _results.append((bool(ok), name, detail))
    return bool(ok)


# --- concept seeds: deterministic, distinct words ----------------------------
_WORDS = ["lantern", "harbour", "quartz", "meadow", "cinder", "orchid",
          "glacier", "ember", "tundra", "falcon", "willow", "basalt"]
_word_i = [0]


def _fake_draw_seeds(n=1, rng=None, away_from=None):
    out = []
    for _ in range(n):
        w = _WORDS[_word_i[0] % len(_WORDS)]
        _word_i[0] += 1
        out.append({"word": w, "token_id": 1000 + _word_i[0],
                    "u32": concept_seed.encode(w),
                    "hex": f"0x{concept_seed.encode(w):08X}"})
    return out


concept_seed.available = lambda: True
concept_seed.draw_seeds = _fake_draw_seeds

# --- the model: a fake upstream ----------------------------------------------
_script: list = []            # answers, in call order; "FAIL" -> HTTP 500
_seen: list[dict] = []        # request bodies, in order
_lane_at_call: list[int] = []  # helper lanes held while each call ran
_live = {"now": 0, "max": 0}
_live_lock = threading.Lock()


def _sse(content: str) -> bytes:
    def ev(delta=None, finish=None):
        return b"data: " + json.dumps({
            "id": "u", "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": delta or {},
                         "finish_reason": finish}]}).encode() + b"\n\n"
    out = [ev({"role": "assistant"})]
    out += [ev({"content": content[i:i + 13]})
            for i in range(0, len(content), 13)]
    out.append(ev({}, "stop"))
    return b"".join(out) + b"data: [DONE]\n\n"


class _Up(BaseHTTPRequestHandler):
    def do_POST(self):                                           # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        with _live_lock:
            _live["now"] += 1
            _live["max"] = max(_live["max"], _live["now"])
            _seen.append(body)
            _lane_at_call.append(admission._helper_stats["inflight"])
            nxt = _script.pop(0) if _script else "FAIL"
        try:
            time.sleep(0.05)          # long enough for an overlap to show
            if nxt == "FAIL":
                self.send_response(500)
                self.end_headers()
                return
            data = _sse(nxt)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        finally:
            with _live_lock:
                _live["now"] -= 1

    def log_message(self, *a):                                   # noqa: D102
        pass


_srv = ThreadingHTTPServer(("127.0.0.1", 0), _Up)
threading.Thread(target=_srv.serve_forever, daemon=True).start()
proxy.UPSTREAM = f"http://127.0.0.1:{_srv.server_address[1]}"

TASK = "Write total_even(nums) in Python: the sum of the even numbers."
PAYLOAD = {"model": "bonsai", "messages": [{"role": "user", "content": TASK}]}


def upstream(*answers: str) -> None:
    _script[:] = list(answers)
    _seen.clear()
    _lane_at_call.clear()
    _live.update(now=0, max=0)


def original(content: str) -> dict:
    return {"content": content,
            "raw": {"choices": [{"message": {"content": content},
                                 "finish_reason": "stop"}]}}


def last_user(body: dict) -> str:
    return next((m.get("content") or "") for m in reversed(body["messages"])
                if m.get("role") == "user")


def fenced(code: str, lang: str = "python", prose: str = "Here it is:",
           closed: bool = True) -> str:
    return f"{prose}\n\n```{lang}\n{code}\n" + ("```\n" if closed else "")


# Two correct, similar solutions and one different correct solution.
LOOP_A = """def total_even(nums):
    total = 0
    for n in nums:
        if n % 2 == 0:
            total += n
    return total
"""
LOOP_B = """def total_even(values):
    # sum the even ones
    acc = 0
    for v in values:
        if v % 2 == 0:
            acc += v
    return acc
"""
OUTLIER = """def total_even(nums):
    return sum(filter(lambda k: not k & 1, nums))
"""
BROKEN = """def total_even(nums):
    return sum(n for n in nums if n % 2 == 0"""
BROKEN_2 = """def total_even(nums)
    return 0"""
TRUNC = """def total_even(nums):
    total = 0"""


def _winner(v: dict) -> str | None:
    return (v.get("winner") or {}).get("variant")


# ======================================================= sequential ========

def test_a_clear_winner_stops_at_two():
    # Only B parses: B wins, no tie-breaker.
    upstream(fenced(LOOP_B))
    v = fanout.run(PAYLOAD, original=original(fenced(BROKEN)), n=3)
    check(len(_seen) == 1 and v["steps"] == 2
          and v["stop_reason"] == "clear: only one parses",
          "only one parses: ONE helper generation, stop after two",
          f"{len(_seen)} {v.get('steps')} {v.get('stop_reason')}")
    check(_winner(v) == "direct" and v["selection"] == "code_grade"
          and v["winner_index"] == 1,
          "and the parsing candidate (B) wins",
          f"{_winner(v)} {v.get('selection')}")
    rows = v["candidates"]
    check(rows[0]["parses"] is False and rows[0]["errors"]
          and rows[1]["parses"] is True and rows[0]["role"] == "main"
          and rows[1]["role"] == "helper",
          "per candidate: role, parses, error count",
          json.dumps(rows)[:400])

    # Both parse and agree (a renamed copy): the main brain's answer stays.
    upstream(fenced(LOOP_B))
    v = fanout.run(PAYLOAD, original=original(fenced(LOOP_A)), n=3)
    check(len(_seen) == 1 and v["steps"] == 2
          and v["stop_reason"] == "clear: agreement"
          and _winner(v) == "original"
          and v["similarity_ab"] >= fanout.AGREE,
          "both parse and agree: A wins at two steps",
          f"{len(_seen)} {v.get('stop_reason')} {v.get('similarity_ab')}")
    check(fanout.AGREE == 0.80, "AGREE is the documented 0.80 choice")

    # And through the proxy: B's code REPLACES nothing yet (that is
    # complete()'s job) but _delivers_winner says it would.
    upstream(fenced(LOOP_B))
    note, rec, win = proxy._fan_out(dict(PAYLOAD, _selection={"fanout_n": 3}),
                                    {"content": fenced(BROKEN)}, "stop")
    check(rec["mode"] == "sequential" and rec["steps"] == 2
          and rec["winner"] == "direct" and rec["n"] == 2
          and rec["asked"] == 3 and proxy._delivers_winner(rec, win),
          "x_yamadori.fanout: mode, steps, n, asked, winner; delivered",
          json.dumps(rec)[:500])


def test_disagreement_runs_the_tiebreaker_with_both_candidates():
    upstream(fenced(OUTLIER), fenced(LOOP_B, prose="Fixed:"))
    v = fanout.run(PAYLOAD, original=original(fenced(LOOP_A)), n=3)
    check(len(_seen) == 2 and v["steps"] == 3
          and v["stop_reason"].startswith("tie-breaker: both parse but "
                                          "disagree"),
          "both parse, similarity below AGREE: the tie-breaker runs",
          f"{len(_seen)} {v.get('stop_reason')} {v.get('similarity_ab')}")
    check(v["similarity_ab"] is not None and v["similarity_ab"] < fanout.AGREE,
          "the A-B similarity is recorded", str(v.get("similarity_ab")))
    prompt = last_user(_seen[1]) if len(_seen) > 1 else ""
    check(TASK in prompt and "total = 0" in prompt
          and "lambda k: not k & 1" in prompt,
          "C's prompt carries the task and BOTH candidates' code",
          prompt[:600])
    check(prompt.count("parses; no syntax errors") == 2
          and "disagree" in prompt,
          "and each one's check result, and that they disagree",
          prompt[:900])
    check([c["variant"] for c in v["candidates"]]
          == ["original", "direct", "tiebreak"]
          and [c["role"] for c in v["candidates"]]
          == ["main", "helper", "helper"],
          "three candidates: A (main), B and C (helper)",
          json.dumps(v["candidates"])[:400])
    # A and C are the same algorithm; C is preferred on the tie.
    check(_winner(v) == "tiebreak" and v["selection"] == "code_medoid"
          and "tie-breaker" in v["winner_reason"],
          "consensus over [A, B, C]: the medoid, C preferred on a tie",
          f"{_winner(v)} {v.get('winner_reason')}")


def test_neither_parses_runs_the_tiebreaker():
    upstream(fenced(BROKEN_2), fenced(LOOP_A))
    v = fanout.run(PAYLOAD, original=original(fenced(BROKEN)), n=3)
    check(len(_seen) == 2 and v["steps"] == 3
          and "neither parses" in v["stop_reason"],
          "neither parses: the tie-breaker runs",
          f"{len(_seen)} {v.get('stop_reason')}")
    prompt = last_user(_seen[1]) if len(_seen) > 1 else ""
    check(prompt.count("does NOT parse: line") == 2
          and "Neither one is a complete solution" in prompt,
          "C is told each candidate's syntax errors, with lines",
          prompt[-900:])
    check(_winner(v) == "tiebreak",
          "the only parsing candidate, C, wins", str(_winner(v)))

    # n=2 leaves no room for C: the original is kept, and it says why.
    upstream(fenced(BROKEN_2))
    v = fanout.run(PAYLOAD, original=original(fenced(BROKEN)), n=2)
    check(len(_seen) == 1 and v["steps"] == 2 and _winner(v) == "original"
          and v["stop_reason"].startswith("unresolved"),
          "fanout 2: A + B only, no tie-breaker",
          f"{len(_seen)} {v.get('stop_reason')}")


def test_at_most_one_helper_context_is_live():
    upstream(fenced(OUTLIER), fenced(LOOP_B))
    fanout.run(PAYLOAD, original=original(fenced(LOOP_A)), n=3)
    check(len(_lane_at_call) == 2 and all(x == 1 for x in _lane_at_call),
          "B and C each ran while the helper lane was held",
          str(_lane_at_call))
    check(_live["max"] == 1, "one generation at a time within a fan-out",
          str(_live["max"]))

    # Two requests fanning out at once: the lane serialises them.
    upstream(fenced(OUTLIER), fenced(LOOP_B), fenced(OUTLIER), fenced(LOOP_B))
    ts = [threading.Thread(target=fanout.run, args=(PAYLOAD,),
                           kwargs={"original": original(fenced(LOOP_A)),
                                   "n": 3}) for _ in range(2)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=30)
    check(len(_seen) == 4 and _live["max"] == 1
          and all(x == 1 for x in _lane_at_call),
          "two concurrent fan-outs: four helper generations, never two at "
          "once", f"{len(_seen)} max_live={_live['max']} {_lane_at_call}")
    check(admission._helper_stats["inflight"] == 0,
          "and the lane is released afterwards")


def test_a_busy_helper_lane_is_recorded_not_raised():
    upstream(fenced(LOOP_B))
    admission._helper_sem.acquire()
    saved = fanout.LANE_WAIT
    fanout.LANE_WAIT = 0.2
    try:
        note, rec, win = proxy._fan_out(
            dict(PAYLOAD, _selection={"fanout_n": 3}),
            {"content": fenced(BROKEN)}, "stop")
    finally:
        fanout.LANE_WAIT = saved
        admission._helper_sem.release()
    check(not _seen and rec and rec.get("skipped") == "helper busy"
          and not proxy._delivers_winner(rec, win),
          "lane busy: nothing generated, skipped 'helper busy', the original "
          "kept", json.dumps(rec)[:300])


def test_the_budget_is_the_helpers_whole_share():
    msgs = [{"role": "user", "content": TASK + " " + "context " * 400}]
    t = tiers.resolve({"reasoning_effort": "high"})
    shaped = tiers.apply({"model": "bonsai", "messages": msgs,
                          "max_tokens": 3000}, t)
    before = json.dumps(shaped, sort_keys=True)
    upstream(fenced(OUTLIER), fenced(LOOP_B))
    fanout.run(shaped, original=original(fenced(LOOP_A)), n=3)
    helper = budget.budgets()["helper"]
    est_b = tiers.estimate_prompt_tokens({"messages": msgs})
    want = tiers.budget(3000, True, role="helper", share_n=1,
                        prompt_tokens=est_b)
    b = _seen[0] if _seen else {}
    check((b.get("max_tokens"), b.get("reasoning_budget_tokens"))
          == (want["max_tokens"], want["reasoning_budget_tokens"])
          and want["reasoning_budget_tokens"] == helper - est_b - 3000,
          "B: thinking = helper share (share_n=1) - prompt - answer",
          f"{(b.get('max_tokens'), b.get('reasoning_budget_tokens'))} vs "
          f"{want}")
    c = _seen[1] if len(_seen) > 1 else {}
    check(c.get("max_tokens", 0) - c.get("reasoning_budget_tokens", 0) == 3000
          and c.get("reasoning_budget_tokens", 0)
          < b.get("reasoning_budget_tokens", 0)
          and c.get("max_tokens", 0) + est_b <= helper,
          "C: the helper share too, re-derived for its longer prompt",
          f"{c.get('max_tokens')} {c.get('reasoning_budget_tokens')}")
    check(all(q.get("temperature") == 1.0 and q.get("top_k") == 20
              for q in _seen),
          "both sample at the vendor settings the shaped payload carries",
          json.dumps([q.get("temperature") for q in _seen]))
    check(json.dumps(shaped, sort_keys=True) == before,
          "the caller's payload is not mutated")


def test_every_second_brain_run_carries_a_distinct_seed():
    upstream(fenced(OUTLIER), fenced(LOOP_B))
    note, rec, win = proxy._fan_out(dict(PAYLOAD, _selection={"fanout_n": 3}),
                                    {"content": fenced(LOOP_A)}, "stop")
    phr = [next((w for w in _WORDS if f"Inspiration word: {w}"
                 in last_user(b)), None) for b in _seen]
    check(len(phr) == 2 and all(phr) and phr[0] != phr[1],
          "B and C each carry a distinct seed phrase in the user turn",
          str(phr))
    seeds = [c.get("seed") for c in rec["candidates"]]
    check(seeds[0] is None and [s and s.get("word") for s in seeds[1:]] == phr
          and all(s and s.get("u32") == concept_seed.encode(s["word"])
                  and s.get("token_id") for s in seeds[1:]),
          "x_yamadori.fanout.candidates[].seed: word and numbers, A none",
          json.dumps(seeds))
    check((concept_seed.last() or {}).get("word") == phr[-1],
          "concept_seed.record() ran: the dashboard's last seed is C's",
          json.dumps(concept_seed.last()))

    # Deep thinking: its one run carries its own seed, recorded too.
    sent: list[dict] = []
    saved = shomen._post

    def fake_post(path, payload, timeout=3600):
        sent.append(json.loads(json.dumps(payload)))
        return {"choices": [{"message": {"content": "It is in core/a.js."},
                             "finish_reason": "stop"}]}
    shomen._post = fake_post
    try:
        payload = {"_selection": {"investigate": True}, "messages": []}
        rec_dt = proxy._drain(proxy._deep_thinking(
            payload, [{"role": "user", "content": "Where is DEFAULT_UP set?"}],
            None, None, None))
    finally:
        shomen._post = saved
    word = ((rec_dt or {}).get("seed") or {}).get("word")
    first = last_user(sent[0]) if sent else ""
    check(word and f"Inspiration word: {word}" in first
          and word not in phr,
          "a deep-thinking run carries its own seed in the user turn",
          f"{word} {first[-120:]}")
    check((rec_dt or {}).get("seed", {}).get("u32")
          == concept_seed.encode(word or ""),
          "and the investigate record carries it (word and number)",
          json.dumps(rec_dt)[:300])


def test_prose_keeps_the_original():
    upstream("It is src/b.ts")
    note, rec, win = proxy._fan_out(dict(PAYLOAD, _selection={"fanout_n": 3}),
                                    {"content": "It is in src/a.ts"}, "stop")
    check(len(_seen) == 1 and rec["steps"] == 2
          and rec["selection"] == "path_consensus"
          and rec["stop_reason"].startswith("prose"),
          "prose: B only, the vote recorded, no tie-breaker",
          f"{len(_seen)} {json.dumps(rec)[:300]}")
    check(not proxy._delivers_winner(rec, win),
          "and the original is what is delivered")
    check("unsettled" in note and rec["votes"] == 2,
          "observed disagreement between A and B is still noted",
          repr(note))


def test_the_proxy_helper_gates():
    msg = {"role": "assistant", "content": "It is in src/a.ts"}

    def payload(n):
        return dict(PAYLOAD, _selection={"fanout_n": n})
    upstream("src/b.ts")
    for args, name in (((payload(1), msg, "stop"), "N=1"),
                       ((payload(3), msg, "length"), "a budget event"),
                       ((payload(3), dict(msg, tool_calls=[{"id": "c"}]),
                         "stop"), "a hand-off of client tool calls"),
                       ((payload(3), {"content": "  "}, "stop"),
                        "an empty answer")):
        note, rec, win = proxy._fan_out(*args)
        check(rec is None and not _seen, f"{name}: nothing runs",
              str(len(_seen)))
    saved = fanout.run
    fanout.run = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    try:
        note, rec, win = proxy._fan_out(payload(3), msg, "stop")
    finally:
        fanout.run = saved
    check(note == "" and rec and rec.get("error") == "RuntimeError",
          "fan-out that raises: answered once, the error recorded",
          json.dumps(rec))


def test_a_failed_second_candidate_keeps_the_original():
    upstream("FAIL", "FAIL")
    v = fanout.run(PAYLOAD, original=original(fenced(LOOP_A)), n=3)
    check(_winner(v) == "original" and v["steps"] == 2
          and v["stop_reason"] == "second candidate failed"
          and v["results"][1].get("error"),
          "B failed: the original is kept and the error recorded",
          f"{v.get('stop_reason')} {v['results'][1].get('error')}")


def test_candidate_code_is_never_executed():
    import builtins
    import subprocess
    sentinel = os.path.join(_TMP, "EXECUTED")
    code = (f"import os\nopen({sentinel!r}, 'w').write('ran')\n"
            f"os.environ['FANOUT_EXECUTED'] = '1'\n"
            f"def total_even(nums):\n    return 0\n")
    tripped: list[str] = []
    saved = {"exec": builtins.exec, "eval": builtins.eval,
             "popen": subprocess.Popen}

    def trip(name):
        def f(*a, **k):
            tripped.append(name)
            return saved[name](*a, **k)
        return f
    upstream(fenced(OUTLIER), fenced(code + "x = 1\n"))
    builtins.exec = trip("exec")
    builtins.eval = trip("eval")
    subprocess.Popen = trip("popen")
    try:
        v = fanout.run(PAYLOAD, original=original(fenced(code)), n=3)
        v2 = fanout.consensus([{"variant": k, "content": fenced(code)}
                               for k in ("a", "b", "c")])
    finally:
        builtins.exec = saved["exec"]
        builtins.eval = saved["eval"]
        subprocess.Popen = saved["popen"]
    check(v["steps"] == 3 and v2["selection"] == "code_medoid",
          "(the sentinel code was graded, tie-broken and judged as code)",
          f"{v.get('steps')} {v2.get('selection')}")
    check(not os.path.exists(sentinel)
          and os.environ.get("FANOUT_EXECUTED") is None,
          "the sentinel was never written: candidate code did not run")
    check(not tripped, "no exec, eval or subprocess during fan-out",
          str(tripped))


# ======================================================= consensus =========

def cands(**kw: str) -> list[dict]:
    return [{"variant": k, "seed": f"s-{k}", "content": v,
             "raw": {"choices": [{"finish_reason": "stop"}]}}
            for k, v in kw.items()]


def test_no_votes_is_not_disagreement():
    v = fanout.consensus(cands(direct="2\n3\n5\n7", evidence="2, 3, 5, 7",
                               skeptical="2 3 5 7"))
    check(v["votes"] == 0 and v["path_votes"] == {} and v["agreement"] is None,
          "no file named: no vote cast, agreement None", json.dumps(
              {"votes": v["votes"], "agreement": v["agreement"]}))
    check(fanout.dissent_note(v) == "", "and no dissent note is written")


def test_observed_disagreement_is_still_reported():
    v = fanout.consensus(cands(direct="It is in src/a.ts",
                               evidence="See src/b.ts line 4",
                               skeptical="Probably src/c.ts"))
    note = fanout.dissent_note(v)
    check(v["agreement"] == 0.33 and "unsettled" in note and "33%" in note,
          "three files, one vote each: reported", repr(note))
    v = fanout.consensus(cands(direct="src/a.ts defines it",
                               evidence="src/a.ts, `sizeKvPool`",
                               skeptical="src/a.ts", terse="src/b.ts"))
    check(v["agreement"] == 0.75 and fanout.dissent_note(v) == ""
          and _winner(v) == "evidence" and v["winner"]["seed"] == "s-evidence",
          "three of four agree: no note; the most-agreed answer wins, whole",
          f"{v['agreement']} {_winner(v)}")


def test_code_the_shortest_broken_or_truncated_answer_does_not_win():
    kw = dict(direct=fenced(LOOP_A), evidence=fenced(LOOP_B, prose="Sure."),
              skeptical=fenced(BROKEN, prose=""))
    v = fanout.consensus(cands(**kw))
    check(min(kw, key=lambda k: len(kw[k])) == "skeptical",
          "(fixture: the broken answer is the shortest)")
    check(v["selection"] == "code_medoid"
          and _winner(v) in ("direct", "evidence")
          and "excluded" in v["winner_reason"],
          "the syntactically broken shortest answer does not win",
          f"{_winner(v)} {v['winner_reason']}")
    rows = {c["variant"]: c for c in v["candidates"]}
    check(rows["skeptical"]["parses"] is False and rows["skeptical"]["errors"]
          and rows["direct"]["parses"],
          "each candidate records whether its code parses, and its errors",
          json.dumps(v["candidates"])[:400])
    v = fanout.consensus(cands(direct=fenced(LOOP_A), evidence=fenced(LOOP_B),
                               skeptical=fenced(TRUNC, prose="",
                                                closed=False)))
    check(_winner(v) in ("direct", "evidence")
          and "truncated" in v["winner_reason"],
          "a truncated answer that parses is set aside", v["winner_reason"])


def test_code_two_that_agree_beat_the_outlier():
    v = fanout.consensus(cands(direct=fenced(LOOP_A),
                               evidence=fenced(OUTLIER, prose=""),
                               skeptical=fenced(LOOP_B)))
    sims = {c["variant"]: c["similarity"] for c in v["candidates"]}
    check(_winner(v) in ("direct", "skeptical")
          and sims["evidence"] < min(sims["direct"], sims["skeptical"]),
          "2 of 3 similar: the medoid is one of the two", json.dumps(sims))


def test_the_preferred_candidate_wins_a_tie():
    rs = cands(original=fenced(LOOP_A), direct=fenced(OUTLIER),
               tiebreak=fenced(LOOP_B))
    v = fanout.consensus(rs, prefer=2)
    check(_winner(v) == "tiebreak" and "tie-breaker" in v["winner_reason"],
          "A and C tie as the medoid: C is preferred", v["winner_reason"])
    v = fanout.consensus(rs)
    check(_winner(v) != "direct", "(without a preference, still not the "
          "outlier)", str(_winner(v)))


def test_code_renamed_variables_agree():
    renamed = LOOP_A.replace("total", "running").replace("nums", "items") \
        .replace(" n ", " item ").replace("n %", "item %") \
        .replace("+= n", "+= item")
    check(fanout.code_similarity(LOOP_A, renamed, "python") == 1.0,
          "python: every name renamed is still full agreement")
    check(fanout.code_similarity(LOOP_A, OUTLIER, "python") < 0.5,
          "and a different algorithm is not")
    check(fanout.code_similarity(LOOP_A, LOOP_B, "python") == 1.0,
          "comments and whitespace do not count")
    s_api = fanout.code_similarity("def f(x):\n    return sorted(x)\n",
                                   "def f(x):\n    return reversed(x)\n",
                                   "python")
    check(s_api < 1.0, "a name the code only USES keeps its spelling",
          str(s_api))
    ta = "function add(a: number, b: number): number {\n  return a + b;\n}"
    tb = ("function plus(x: number, y: number): number {\n"
          "  // add them\n  return x + y;\n}")
    tc = "function add(a: number, b: number): number {\n  return a * b;\n}"
    check(fanout.code_similarity(ta, tb, "typescript") == 1.0
          and fanout.code_similarity(ta, tc, "typescript") < 1.0,
          "typescript: renamed agrees, a changed operator does not")


def test_no_code_answers_still_use_path_consensus():
    v = fanout.consensus(cands(direct="src/a.ts defines it",
                               evidence="src/a.ts:\n```ts\nexport const "
                                        "sizeKvPool = 1;\n```",
                               skeptical="src/a.ts, `sizeKvPool`"))
    check(v["selection"] == "path_consensus",
          "a minority code snippet leaves the path vote in charge")
    v = fanout.consensus(cands(direct="```\n2\n3\n5\n7\n```",
                               evidence="```\n2 3 5 7\n```",
                               skeptical="2, 3, 5, 7"))
    check(v["selection"] == "path_consensus",
          "untagged output blocks are not taken for code")


def test_nothing_parses_falls_back_to_the_longest_and_says_so():
    kw = dict(direct=fenced(BROKEN, prose=""),
              evidence=fenced(BROKEN + "\n# longer", prose="Explained. " * 5),
              skeptical=fenced("def total_even(nums) return 0"))
    v = fanout.consensus(cands(**kw))
    check(v["selection"] == "fallback" and v["fallback"] is True
          and _winner(v) == max(kw, key=lambda k: len(kw[k]))
          and "longest" in v["winner_reason"],
          "no candidate parses: 'fallback', the longest, and why",
          f"{v.get('selection')} {_winner(v)}")
    check(not proxy._delivers_winner(dict(selection=v["selection"]),
                                     v["winner"]),
          "a fallback is recorded, never delivered")


def test_the_selection_record_is_compact():
    upstream(fenced(OUTLIER), fenced(LOOP_B))
    v = fanout.run(PAYLOAD, original=original(fenced(LOOP_A)), n=3)
    rec = fanout.selection_record(v)
    check({"mode", "steps", "stop_reason", "similarity_ab", "selection",
           "winner_index", "why", "candidates"} <= set(rec)
          and rec["mode"] == "sequential" and rec["steps"] == 3
          # seed_echo_stripped: #13 (SELF-IMPROVEMENT-LOG), a count.
          and all(set(c) == {"variant", "role", "seed", "parses",
                             "check_mode", "errors", "similarity", "length",
                             "seed_echo_stripped"}
                  for c in rec["candidates"]),
          "x_yamadori.fanout: mode, steps, stop reason, A-B similarity, and "
          "per candidate role/seed/parses/check_mode/errors/similarity/length"
          "/seed_echo_stripped",
          json.dumps(rec)[:500])
    check("total_even" not in json.dumps(rec) and len(json.dumps(rec)) < 1200,
          "and carries no answer text", str(len(json.dumps(rec))))


# ================================================ the question's code =======
# THE GENERAL RULE (operator, 2026-09-23): when the question carries fenced
# code, a candidate parses if it parses on its own OR appended after one of
# those blocks. No phrasing is read. The skeletons are
# mcp/test_code_check.py's HEADER_* (lb-20260923 eae49210) and LOOP_*
# (3cd8c16a), reduced, copied: importing that module would start its servers.
HEADER_PREFIX = (
    "class Solution(object):\n"
    "    def minimumAddedCoins(self, coins, target):\n"
    "        coins.sort()\n"
    "        result = reachable = 0\n"
    "        for x in coins:\n"
    "            while not reachable >= x-1:\n"
    "                result += 1\n"
    "                reachable += reachable+1\n"
    "            reachable += x\n"
    "        while not reachable >= target:")
HEADER_MODEL = (
    "    result += 1\n"
    "    reachable += reachable + 1\n"
    "return result")
HEADER_RIGHT = (
    "            result += 1\n"
    "            reachable += reachable + 1\n"
    "        return result")
HEADER_RIGHT_2 = (
    "            reachable += reachable + 1\n"
    "            result += 1\n"
    "        return result")
LOOP_PREFIX = (
    "class Solution(object):\n"
    "    def earliestSecondToMarkIndices(self, nums, changeIndices):\n"
    "        def check(t):\n"
    "            return t >= len(nums)\n"
    "\n"
    "        left, right = sum(nums)+len(nums), len(changeIndices) \n"
    "        while left <= right:\n"
    "            mid = left+(right-left)//2")
LOOP_RIGHT = (
    "            if check(mid):\n"
    "                right = mid - 1\n"
    "            else:\n"
    "                left = mid + 1\n"
    "        if left > len(changeIndices):\n"
    "            return -1\n"
    "        return left")
# A whole-class rewrite: the shape the tie-breaker delivered in 6 of 10
# completion rows on 2026-09-23 (5 scored). It parses on its own.
LOOP_REWRITE = (LOOP_PREFIX.replace("len(changeIndices) \n",
                                    "len(changeIndices)\n")
                + "\n" + LOOP_RIGHT)


def completion_task(prefix: str) -> str:
    """The SHAPE of a LiveBench coding_completion prompt (paraphrased, as in
    test_code_check.completion_question)."""
    return ("### Instructions: complete the program.\n### Question:\n"
            "(problem statement)\n\n### Format: You will use the following "
            "starter code.\n```python\n" + prefix + "\n```\n\n### Answer: "
            "(only the missing portion)\n")


def casual_task(prefix: str) -> str:
    return "finish this function\n\n```python\n" + prefix + "\n```"


def task(text: str) -> dict:
    return {"model": "bonsai", "messages": [{"role": "user", "content": text}]}


def test_candidates_parse_alone_or_appended_to_the_question():
    # On its own -- all fan-out did before -- neither continuation parses,
    # so the grade could not separate them (lb-20260923-minp0: 10 of 10
    # completion rows went to the tie-breaker as "neither parses").
    check(fanout.check_code(HEADER_MODEL, "python")["parses"] is False
          and fanout.check_code(HEADER_RIGHT, "python")["parses"] is False,
          "(on its own, the broken and the correct continuation both fail)")

    for label, text in (("LiveBench phrasing", completion_task(HEADER_PREFIX)),
                        ("'finish this function'", casual_task(HEADER_PREFIX))):
        upstream(fenced(HEADER_RIGHT))
        v = fanout.run(task(text), original=original(fenced(HEADER_MODEL)),
                       n=3)
        rows = v["candidates"]
        check(len(_seen) == 1 and v["stop_reason"] == "clear: only one parses"
              and _winner(v) == "direct"
              and [r["parses"] for r in rows] == [False, True]
              and [r["check_mode"] for r in rows] == ["neither", "appended"],
              f"{label}: only B parses (appended); B wins at two steps",
              f"{len(_seen)} {v.get('stop_reason')} {json.dumps(rows)[:300]}")
    rec = fanout.selection_record(v)
    check([c.get("check_mode") for c in rec["candidates"]]
          == ["neither", "appended"],
          "x_yamadori.fanout.candidates[].check_mode", json.dumps(rec)[:400])

    # A whole rewrite parses on its own: "standalone", no warning needed.
    upstream(fenced(LOOP_REWRITE))
    v = fanout.run(task(completion_task(LOOP_PREFIX)),
                   original=original(fenced(LOOP_RIGHT)), n=2)
    check([r["parses"] for r in v["candidates"]] == [True, True]
          and [r["check_mode"] for r in v["candidates"]]
          == ["appended", "standalone"],
          "a correct continuation (appended) and a whole rewrite "
          "(standalone) both parse", json.dumps(v["candidates"])[:400])

    # Neither parses either way: the tie-breaker's prompt is the one it
    # always was (operator: its whole-class rewrites score).
    upstream(fenced("    result += 1\n  return result"), fenced(HEADER_RIGHT))
    v = fanout.run(task(completion_task(HEADER_PREFIX)),
                   original=original(fenced(HEADER_MODEL)), n=3)
    prompt = last_user(_seen[1]) if len(_seen) > 1 else ""
    check(v["steps"] == 3 and "neither parses" in v["stop_reason"]
          and prompt.count("does NOT parse: line") == 2
          and "Neither one is a complete solution that parses as written"
          in prompt
          and "any logic error you find), and give the complete python code "
              "in one fenced block." in prompt,
          "neither parses: C's prompt unchanged", prompt[-700:])
    check(_winner(v) == "tiebreak"
          and v["candidates"][2]["check_mode"] == "appended",
          "C, a correct continuation, parses appended and wins",
          json.dumps(v["candidates"])[:400])

    # Consensus with the question's code: a broken one is excluded.
    pc = fanout._prompt_code(task(completion_task(HEADER_PREFIX))["messages"])
    v = fanout.consensus(cands(a=fenced(HEADER_RIGHT), b=fenced(HEADER_MODEL),
                               c=fenced(HEADER_RIGHT_2)), prompt=pc)
    check(v["selection"] == "code_medoid" and _winner(v) in ("a", "c")
          and {c["variant"]: c["parses"] for c in v["candidates"]}
          == {"a": True, "b": False, "c": True},
          "consensus: the one that parses neither way is excluded",
          json.dumps(v["candidates"])[:400])


def test_other_tasks_are_graded_as_before():
    # A code task with no code in the question: on its own, check_mode None,
    # and C's prompt unchanged.
    upstream(fenced(BROKEN_2), fenced(LOOP_A))
    v = fanout.run(PAYLOAD, original=original(fenced(BROKEN)), n=3)
    prompt = last_user(_seen[1]) if len(_seen) > 1 else ""
    check(all(c["check_mode"] is None for c in v["candidates"])
          and [c["parses"] for c in v["candidates"]] == [False, False, True],
          "no code in the question: every candidate on its own, check_mode "
          "None, as before", json.dumps(v["candidates"])[:400])
    check("any logic error you find), and give the complete python code in "
          "one fenced block." in prompt
          and "Neither one is a complete solution that parses as written"
          in prompt and "appended" not in prompt,
          "and C's prompt is the one it always was", prompt[-300:])
    # Prose: no mode at all.
    upstream("It is src/b.ts")
    v = fanout.run(PAYLOAD, original=original("It is in src/a.ts"), n=3)
    check(all(c.get("check_mode") is None
              for c in fanout.selection_record(v)["candidates"]),
          "prose: check_mode is None", json.dumps(fanout.selection_record(v))
          [:300])


def test_variants_sample_at_the_vendor_settings():
    check(all("temperature" not in v for v in fanout.VARIANTS)
          and fanout.VARIANTS[0]["name"] == "direct",
          "variants carry no temperature of their own; B is 'direct'",
          json.dumps(fanout.VARIANTS)[:200])


TESTS = [test_variants_sample_at_the_vendor_settings,
         test_a_clear_winner_stops_at_two,
         test_disagreement_runs_the_tiebreaker_with_both_candidates,
         test_neither_parses_runs_the_tiebreaker,
         test_at_most_one_helper_context_is_live,
         test_a_busy_helper_lane_is_recorded_not_raised,
         test_the_budget_is_the_helpers_whole_share,
         test_every_second_brain_run_carries_a_distinct_seed,
         test_prose_keeps_the_original,
         test_the_proxy_helper_gates,
         test_a_failed_second_candidate_keeps_the_original,
         test_candidate_code_is_never_executed,
         test_no_votes_is_not_disagreement,
         test_observed_disagreement_is_still_reported,
         test_code_the_shortest_broken_or_truncated_answer_does_not_win,
         test_code_two_that_agree_beat_the_outlier,
         test_the_preferred_candidate_wins_a_tie,
         test_code_renamed_variables_agree,
         test_no_code_answers_still_use_path_consensus,
         test_nothing_parses_falls_back_to_the_longest_and_says_so,
         test_the_selection_record_is_compact,
         test_candidates_parse_alone_or_appended_to_the_question,
         test_other_tasks_are_graded_as_before]


def test_the_hand_back_names_only_what_differs():
    """fanout.handback (operator decision, 2026-09-23): what the other
    candidates say that the delivered answer does not. The end-to-end
    delivery is in mcp/test_fanout_delivery.py."""
    def code(results, winner):
        return fanout.handback({"results": [{"content": fenced(c)}
                                            for c in results],
                                "code_language": "python",
                                "winner_index": winner})
    hb = code([LOOP_A, OUTLIER], 0)
    check(hb and hb["kind"] == "code" and hb["from"] == ["B"]
          and "`filter`, `sum`" in hb["text"],
          "code: the loser's APIs the winner does not use", str(hb))
    check(code([LOOP_A, LOOP_B], 0) is None,
          "renamed variables are not a difference: nothing handed back")
    check(code([LOOP_A, BROKEN_2], 0) is None,
          "a loser whose code does not parse names nothing (its binding "
          "sites are error nodes)")
    check(code([OUTLIER, LOOP_A], 0) is None,
          "a loser that uses nothing the winner lacks: not material")
    pro = fanout.handback({"results": [
        {"content": "Cache the plan in the loader. It is simple."},
        {"content": "Cache the plan in the loader. A bounded LRU keeps "
                    "memory flat when the workload has many distinct plans."},
        {"content": "See `PlanCache` for the existing cache."}]})
    check(pro and pro["kind"] == "prose" and pro["from"] == ["B", "C"]
          and "(B) A bounded LRU" in pro["text"]
          and "(C) See `PlanCache`" in pro["text"]
          and "Cache the plan in the loader." not in pro["text"],
          "prose: each candidate's new points, labelled by candidate; "
          "shared points left out", str(pro))
    long = fanout.handback({"results": [
        {"content": "Short."},
        {"content": " ".join(f"Point {i} names widget{i} gadget{i} "
                             f"sprocket{i} flange{i} bracket{i}."
                             for i in range(20))}]})
    check(long and long["text"].count("\n- ") + 1
          <= fanout.ALT_MAX_POINTS + 1
          and len(long["text"]) <= fanout.ALT_MAX_CHARS + 120
          and "more points not shown" in long["text"],
          "prose hand-back is capped, and says so", long["text"][-160:])


TESTS.append(test_the_hand_back_names_only_what_differs)


def test_the_seed_is_not_copied_into_delivered_code():
    """#13 (docs/SELF-IMPROVEMENT-LOG.md). Live gate 2026-09-24: C's seed was
    `humanidad` and the delivered `is_balanced` opened with `# humanidad`.
    (1) Every second-brain job's user message says what the word is for and
    that it is not part of the answer. (2) A leading line that only repeats
    the seed -- the answer's first line, or a fenced block's -- is removed
    from a candidate and from a fix-up's code, and counted; a comment that
    says more, or sits elsewhere, is the model's and stays."""
    f = fanout.strip_seed_echo
    blk = "Here:\n\n```python\n# humanidad\ndef f():\n    return 1\n```\n"
    out, k = f(blk, "humanidad")
    check(k == 1 and "# humanidad" not in out and "def f():" in out,
          "the live shape: a block opening `# humanidad` loses that line",
          repr(out))
    for text, word, want in (
            ("```js\n// Humanidad.\nconst a = 1;\n```", "humanidad", 1),
            ("# Humanidad\n\n```py\nx = 1\n```", "humanidad", 1),
            ("```py\n# humanidad is the theme here\nx = 1\n```",
             "humanidad", 0),
            ("```py\nx = 1\n# humanidad\n```", "humanidad", 0),
            ("```py\n#!/usr/bin/env python\nx = 1\n```", "python", 0),
            ("```rust\n/* cedar */\nfn main() {}\n```", "cedar", 1),
            (blk, None, 0)):
        _, k = f(text, word)
        check(k == want, f"strip_seed_echo({text[:34]!r}..., {word!r}) "
                         f"removes {want}", str(k))
    # (1) the wording, through the real fan-out: B carries it. (2) B copied
    #     its seed as the block's first line (the live shape): removed from
    #     the candidate that is delivered.
    pin = {"word": "falcon", "token_id": 7,
           "u32": concept_seed.encode("falcon")}
    upstream("```python\n# falcon\n" + LOOP_B + "```\n")
    v = fanout.run(PAYLOAD, original=original(fenced(BROKEN)), n=3,
                   seed_for=lambda job, prompt: dict(pin))
    user = last_user(_seen[0]) if _seen else ""
    check("Inspiration word: falcon" in user
          and "it is not part of the answer" in user
          and "shape how you approach" in user,
          "B's user message names the seed as inspiration for the approach, "
          "not part of the answer", user[-200:])
    b = (v.get("results") or [{}, {}])[1]
    w = (v.get("winner") or {}).get("content") or ""
    rows = fanout.selection_record(v).get("candidates") or [{}, {}]
    seed_b = (b.get("seed") or "")
    check(_winner(v) == "direct" and f"# {seed_b}" not in w
          and b.get("seed_echo_stripped") == 1
          and rows[1].get("seed_echo_stripped") == 1,
          "the winning B arrives without its seed line, and "
          "x_yamadori.fanout.candidates[].seed_echo_stripped counts it",
          json.dumps({"seed": seed_b, "head": w[:60], "rows": rows})[:400])
    # (3) the fix-up job: its code is written into the client's file.
    saved = shomen._post
    shomen._post = lambda path, payload, timeout=3600: {
        "choices": [{"message": {"content": "```python\n# cedar\n"
                                            "def f():\n    return 1\n```"},
                     "finish_reason": "stop"}]}
    try:
        job = shomen.fixup([{"path": "hi.py", "language": "python",
                             "kind": "file", "code": "def f(:\n    return 1\n",
                             "errors": [{"line": 1, "col": 7,
                                         "message": "invalid syntax"}]}],
                           "Write hi.py", check=lambda u, c: [],
                           seed={"word": "cedar", "token_id": 1, "u32": 1})
    finally:
        shomen._post = saved
    u = (job.get("units") or [{}])[0]
    check(u.get("code", "").startswith("def f():")
          and u.get("seed_echo_stripped") == 1,
          "a fix-up's code arrives without the seed line, counted",
          json.dumps(u)[:300])


TESTS.append(test_the_seed_is_not_copied_into_delivered_code)


def main() -> int:
    for fn in TESTS:
        print(f"\n--- {fn.__name__} ---")
        n0 = len(_results)
        try:
            fn()
        except Exception:                                        # noqa: BLE001
            check(False, f"{fn.__name__} itself raised",
                  traceback.format_exc().strip()[-800:])
        for ok, name, detail in _results[n0:]:
            print(("  pass  " if ok else "  FAIL  ") + name
                  + (f"   <- {detail}" if not ok and detail else ""))
    passed = sum(1 for ok, _, _ in _results if ok)
    total = len(_results)
    print(f"\n{'=' * 70}\n  {passed}/{total} checks passed")
    _srv.shutdown()
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
