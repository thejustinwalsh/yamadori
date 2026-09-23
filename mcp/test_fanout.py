#!/usr/bin/env python
"""Fan-out's vote and its note, and the proxy's one fan-out helper. No GPU.

WHAT THIS IS GATING -- docs/SELECTION-BUILD.md, harm 2

  1. NO VOTES IS NOT DISAGREEMENT. An answer naming no file cast no path vote,
     `run()` reported agreement 0.0, and `dissent_note` appended "only 0%
     agreed on the same file. Treat this as unsettled." to every such answer
     at `high` and `max` -- the live suite's 25-primes answer included.
     Agreement over nothing is now None and says nothing.
  2. Disagreement that WAS observed is still reported, with the alternatives.
  3. The winning variant is KEPT (FINDINGS #18: computed, then dropped).
  4. `proxy._fan_out` runs only on a finished text answer and only when the
     selection engine chose N > 1; a failed fan-out answers once, never 500s.
  5. n samples run at once in the conversation's half of the pool, so `run()`
     re-derives each sample's budget with tiers.rebudget(share_n=n) -- a
     fan-out must not overflow the share it came from (mcp/budget.py, the
     split of 2026-09-22).

Each variant's generation is stubbed at `fanout._one`, so what is tested is
the vote, the note and the gating -- not the model.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

_TMP = tempfile.mkdtemp(prefix="yamadori_test_fanout_")
os.environ["YAMADORI_CORPUS_DB"] = os.path.join(_TMP, "corpus.sqlite3")
os.environ["YAMADORI_NEBARI_DB"] = os.path.join(_TMP, "nebari.sqlite3")
os.environ["RINGS_DB"] = os.path.join(_TMP, "rings.sqlite3")
os.environ["YAMADORI_PKG_DIR"] = os.path.join(_TMP, "pkgs")
os.environ["CODE_INDEX_DB"] = os.path.join(_TMP, "code.sqlite3")
# Nothing here may reach a model. A reserved port refuses at once.
os.environ["LLAMA_STACK_URL"] = "http://127.0.0.1:1"

import budget  # noqa: E402
import fanout  # noqa: E402
import proxy  # noqa: E402
import tiers  # noqa: E402

# budget.pool_size() would ask the live server. Pinned to the shipped `-c`.
budget._POOL = 147456
tiers._accepted = tiers.FALLBACK_EFFORTS     # no network for the template

_results: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> bool:
    _results.append((bool(ok), name, detail))
    return bool(ok)


ANSWERS: dict[str, str] = {}
_calls: list[str] = []
_payloads: list[dict] = []


def _fake_one(payload: dict, variant: dict, timeout: int) -> dict:
    _calls.append(variant["name"])
    _payloads.append(payload)
    text = ANSWERS.get(variant["name"], "")
    if text == "RAISE":
        raise RuntimeError("upstream died")
    return {"variant": variant["name"], "seed": variant.get("seed"),
            "content": text, "raw": {}}


fanout._one = _fake_one
fanout._seeds = lambda payload, n: ["lantern", "harbour", "quartz", "meadow"][:n]

PAYLOAD = {"model": "bonsai", "messages": [{"role": "user", "content": "q"}]}


def answers(**kw: str) -> None:
    ANSWERS.clear()
    ANSWERS.update(kw)
    _calls.clear()
    _payloads.clear()


# ---------------------------------------------------------------------------


def test_no_votes_is_not_disagreement():
    answers(direct="2\n3\n5\n7", evidence="2, 3, 5, 7", skeptical="2 3 5 7")
    v = fanout.run(PAYLOAD, n=3)
    check(v["n"] == 3, "three answers came back", str(v.get("n")))
    check(v["votes"] == 0 and v["path_votes"] == {},
          "an answer that names no file casts no path vote",
          json.dumps(v["path_votes"]))
    check(v["agreement"] is None,
          "agreement over nothing is None, not 0.0", repr(v["agreement"]))
    note = fanout.dissent_note(v)
    check(note == "", "and no dissent note is written", repr(note))
    check("unsettled" not in note and "0%" not in note,
          "(the exact text the live primes answer carried is gone)")


def test_observed_disagreement_is_still_reported():
    answers(direct="It is in src/a.ts", evidence="See src/b.ts line 4",
            skeptical="Probably src/c.ts")
    v = fanout.run(PAYLOAD, n=3)
    check(v["votes"] == 3 and v["agreement"] == 0.33,
          "three different files: one vote each, agreement 1/3",
          json.dumps({"votes": v["votes"], "agreement": v["agreement"]}))
    note = fanout.dissent_note(v)
    check("unsettled" in note and "33%" in note,
          "real disagreement is reported", repr(note))
    check("src/b.ts" in note or "src/c.ts" in note,
          "with the other candidates named", repr(note))


def test_agreement_is_silent_and_the_winner_is_kept():
    answers(direct="src/a.ts defines it", evidence="src/a.ts, `sizeKvPool`",
            skeptical="src/a.ts", terse="src/b.ts")
    v = fanout.run(PAYLOAD, n=4)
    check(v["agreement"] == 0.75 and fanout.dissent_note(v) == "",
          "three of four on one file (0.75): no note",
          json.dumps({"agreement": v["agreement"]}))
    check(v["winner"] and v["winner"]["variant"] == "evidence",
          "the winner is the answer naming the most agreed claims",
          str((v.get("winner") or {}).get("variant")))
    check(v["winner"].get("seed") == "harbour",
          "and it keeps the seed it was written under",
          str(v["winner"].get("seed")))


def test_a_variant_that_fails_does_not_sink_the_rest():
    answers(direct="src/a.ts", evidence="RAISE", skeptical="src/a.ts")
    v = fanout.run(PAYLOAD, n=3)
    check(v["n"] == 2 and any(r.get("error") for r in v["results"]),
          "the failure is recorded, the other two are counted",
          json.dumps([r.get("error") for r in v["results"]]))
    answers(direct="RAISE", evidence="RAISE")
    v = fanout.run(PAYLOAD, n=2)
    check(v["agreement"] is None and v["winner"] is None
          and fanout.dissent_note(v) == "",
          "all failed: no agreement, no winner, no note", json.dumps(
              {k: v.get(k) for k in ("agreement", "winner")}))


def test_the_proxy_helper_gates_and_keeps_the_winner():
    msg = {"role": "assistant", "content": "It is in src/a.ts"}

    def payload(n):
        return dict(PAYLOAD, _selection={"fanout_n": n})

    answers(direct="src/a.ts", evidence="src/b.ts", skeptical="src/c.ts")
    note, rec, win = proxy._fan_out(payload(1), msg, "stop")
    check((note, rec, win) == ("", None, None) and not _calls,
          "N=1: nothing runs", str(_calls))
    note, rec, win = proxy._fan_out(payload(3), msg, "length")
    check(rec is None and not _calls,
          "a budget event is not fanned out", str(_calls))
    note, rec, win = proxy._fan_out(
        payload(3), dict(msg, tool_calls=[{"id": "c"}]), "stop")
    check(rec is None and not _calls,
          "a hand-off of client tool calls is not fanned out", str(_calls))
    note, rec, win = proxy._fan_out(payload(3), {"content": "  "}, "stop")
    check(rec is None and not _calls, "an empty answer is not fanned out")

    note, rec, win = proxy._fan_out(payload(3), msg, "stop")
    check(len(_calls) == 3 and "unsettled" in note,
          "N=3 on a finished answer: three variants, disagreement noted",
          repr(note))
    check(rec["seeds"] == ["lantern", "harbour", "quartz"]
          and rec["asked"] == 3 and rec["n"] == 3 and rec["votes"] == 3,
          "the record carries n, the seeds and the votes", json.dumps(rec))
    check(win and win.get("content") and rec["winner"] == win["variant"],
          "the winning variant comes back whole", json.dumps(rec))
    check(len(json.dumps(rec)) < 400 and "src/a.ts" not in json.dumps(rec),
          "the record itself carries no answer text", json.dumps(rec))

    answers(direct="RAISE")
    saved = fanout.run
    fanout.run = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    try:
        note, rec, win = proxy._fan_out(payload(3), msg, "stop")
    finally:
        fanout.run = saved
    check(note == "" and rec and rec.get("error") == "RuntimeError",
          "fan-out that raises: answered once, the error recorded",
          json.dumps(rec))


def test_each_sample_gets_its_share_of_the_half():
    msgs = [{"role": "user", "content": "where is the pool sized? " * 200}]
    t = tiers.resolve({"reasoning_effort": "high"})
    shaped = tiers.apply({"model": "bonsai", "messages": msgs,
                          "max_tokens": 3000}, t)
    before = json.dumps(shaped, sort_keys=True)
    est = tiers.estimate_prompt_tokens({"messages": msgs})
    main = budget.budgets()["main"]
    check(shaped["reasoning_budget_tokens"] == main - est - 3000,
          "the shaped payload starts with the whole main window",
          str(shaped["reasoning_budget_tokens"]))
    for n in (2, 3, 4):
        answers(direct="src/a.ts", evidence="src/a.ts", skeptical="src/a.ts",
                terse="src/a.ts")
        fanout.run(shaped, n=n)
        want = tiers.budget(3000, True, role="main", share_n=n,
                            prompt_tokens=est)
        got = [(q.get("max_tokens"), q.get("reasoning_budget_tokens"))
               for q in _payloads]
        check(len(_payloads) == n and all(
                  g == (want["max_tokens"], want["reasoning_budget_tokens"])
                  for g in got),
              f"n={n}: every sample is rebudgeted with share_n={n} "
              f"(thinking {want['reasoning_budget_tokens']})", str(got))
        check(all(q["max_tokens"] - q["reasoning_budget_tokens"] == 3000
                  for q in _payloads),
              f"n={n}: and keeps the 3,000-token answer allowance")
        check(sum(q["max_tokens"] + est for q in _payloads) <= main,
              f"n={n}: together the samples fit the main half",
              str(sum(q["max_tokens"] + est for q in _payloads)))
    check(json.dumps(shaped, sort_keys=True) == before,
          "the caller's payload is not mutated by the rebudget")
    answers(direct="src/a.ts", evidence="src/a.ts", skeptical="src/a.ts")
    fanout.run(PAYLOAD, n=3)
    check(all("reasoning_budget_tokens" not in q for q in _payloads),
          "an unshaped payload is passed through, not given a budget",
          str(_payloads[:1]))


def main() -> int:
    for fn in (test_no_votes_is_not_disagreement,
               test_observed_disagreement_is_still_reported,
               test_agreement_is_silent_and_the_winner_is_kept,
               test_a_variant_that_fails_does_not_sink_the_rest,
               test_the_proxy_helper_gates_and_keeps_the_winner,
               test_each_sample_gets_its_share_of_the_half):
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
