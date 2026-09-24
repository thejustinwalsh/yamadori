#!/usr/bin/env python
"""E1, the embedding-head classifier (mcp/e1.py). No GPU, no network.

    python mcp/test_e1.py      -> "N/M checks passed"

WHAT THIS IS GATING (docs/E1.md):

  1. THE CONTRACT: render() is laya_head.render_state("route_in") byte for
     byte; the numpy fit is deterministic and matches scripts/train_laya's
     kfold; a head artefact round-trips (weights hashed), refuses to serve
     below MIN_TRAIN_N or with a different feature contract, and every
     version is kept, promotable and revertible.
  2. ONE EMBEDDING: a request is embedded once and cached (memory and the
     store); a dead embedder is None + a reason, never a guess.
  3. THE PROXY PATH: with YAMADORI_E1=1 selection's second signal is E1 and
     NOTHING calls Laya (selection, skill_select, shomen, fanout); with it
     off, Laya is still the second signal.
  4. PHASE 0.6: an untrained head leaves the rule deciding; a trained
     escalate / route_in head decides its trigger and says so; the embedded
     state is recorded with the row and never reported.
  5. SELF-TUNING: e1.learn promotes a candidate only on a paired win on rows
     neither model trained on, keeps (and records) a loser, embeds nothing,
     and the dashboard shows every version with its n.

Heads here are synthetic (random unit vectors, 1024-d) in a temporary
E1_DIR; the real route_in v1 and its numbers are bench/e1/eval_e1.py's.
"""
from __future__ import annotations

import importlib
import json
import os
import sys
import tempfile
import time
import traceback
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

_TMP = tempfile.mkdtemp(prefix="yamadori_test_e1_")
os.environ["YAMADORI_E1_DIR"] = os.path.join(_TMP, "e1")
os.environ["YAMADORI_CORPUS_DB"] = os.path.join(_TMP, "corpus.sqlite3")
os.environ["YAMADORI_NEBARI_DB"] = os.path.join(_TMP, "nebari.sqlite3")
os.environ["YAMADORI_JOBS_DB"] = os.path.join(_TMP, "jobs.sqlite3")
os.environ["RINGS_DB"] = os.path.join(_TMP, "rings.sqlite3")
os.environ["LLAMA_STACK_URL"] = "http://127.0.0.1:9"
os.environ["LAYA_URL"] = "http://127.0.0.1:9"
os.environ.pop("YAMADORI_E1", None)
for k in ("YAMADORI_STRUGGLE_THRESHOLD", "YAMADORI_KICKOFF_TOKENS",
          "YAMADORI_UNSEEN_PACKAGES", "YAMADORI_SEEN_PACKAGES"):
    os.environ.pop(k, None)

import numpy as np  # noqa: E402

import e1  # noqa: E402

_results: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> bool:
    _results.append((bool(ok), name, detail))
    return bool(ok)


DIM = 1024
_rng = np.random.default_rng(7)
_CALLS: list[str] = []
_VEC: dict[str, np.ndarray] = {}


def _unit(v: np.ndarray) -> np.ndarray:
    return (v / np.linalg.norm(v)).astype(np.float32)


_DIRS = np.stack([_unit(_rng.normal(size=DIM)) for _ in range(3)])
WORDS = ("ESCALATE", "INVESTIGATE", "CLARIFY")


def fake_embed(texts: list[str]) -> np.ndarray:
    """Deterministic per text; a text naming a class word is pulled along
    that word's direction (spread over every dimension, as a real embedding
    carries meaning), everything else is isotropic noise."""
    out = []
    for t in texts:
        _CALLS.append(t)
        if t not in _VEC:
            base = _rng.normal(size=DIM)
            for i, word in enumerate(WORDS):
                if word in t:
                    base += 60.0 * _DIRS[i]
            _VEC[t] = _unit(base)
        out.append(_VEC[t])
    return np.stack(out)


e1._embed = fake_embed


class _NoNetwork(Exception):
    pass


_REAL_URLOPEN = urllib.request.urlopen
_OPENED: list[str] = []


def _guard_urlopen(req, *a, **kw):
    url = getattr(req, "full_url", req)
    _OPENED.append(str(url))
    raise _NoNetwork(f"network call in an offline test: {url}")


urllib.request.urlopen = _guard_urlopen


def _flag(on: bool) -> None:
    if on:
        os.environ["YAMADORI_E1"] = "1"
    else:
        os.environ.pop("YAMADORI_E1", None)


def _save_head(head: str, W, b, n: int, per: dict, status="promoted") -> int:
    v = e1.save_version(head, np.asarray(W, np.float32),
                        np.asarray(b, np.float32),
                        trained_on={"n": n, "per_label": per, "sources": {}},
                        cv={"wd": 0.01, "accuracy": 0.9, "by_wd": {},
                            "folds": 5, "seed": 0},
                        seed=0, author="test", evaluation={}, keys=[],
                        parent=None, status=status)
    e1.promote(head, v, "test", author="test")
    return v


# ================================================================ 1 ========
def test_the_contract():
    lh = importlib.import_module("laya_head")
    for rec in ({"question": "  Where is X defined? ", "context": ""},
                {"question": "q", "context": " we use three 0.185 "},
                {"question": "", "context": None}):
        check(e1.render(rec.get("question"), rec.get("context"))
              == lh.render_state("route_in", rec),
              f"render == laya_head.render_state for {rec!r}")
    tl = importlib.import_module("train_laya") if os.path.exists(
        os.path.join(HERE, "..", "scripts", "train_laya.py")) else None
    y = [0, 1, 2, 0, 1, 2, 0, 0, 1, 2, 1, 0, 2, 2, 1, 0, 0, 1]
    if tl is not None:
        for seed in (0, 3):
            check(e1.kfold(y, 5, seed)
                  == tl.kfold(y, list(range(len(y))), 5, seed, None),
                  f"kfold reproduces scripts/train_laya.kfold (seed {seed})")
    X = _rng.normal(size=(60, 16)).astype(np.float32)
    yy = [int(x[0] > 0) for x in X]
    W1, b1 = e1.fit_logistic(X, yy, 2, 0.01)
    W2, b2 = e1.fit_logistic(X, yy, 2, 0.01)
    check(np.array_equal(W1, W2) and np.array_equal(b1, b2),
          "the fit is deterministic (zero init, full batch)")
    acc = float((e1.probs(W1, b1, X).argmax(1) == np.asarray(yy)).mean())
    check(acc >= 0.95, "the fit separates a separable problem", f"{acc}")
    P = e1.probs(W1, b1, X)
    check(np.allclose(P.sum(1), 1.0, atol=1e-5), "probabilities sum to 1")
    check(e1.WD_GRID == (1e-3, 1e-2),
          "the self-tuner's decay grid is the two stable decays")
    # artefact round trip, contract, refusal below MIN_TRAIN_N
    labels = e1.HEADS["route_in"]["labels"]
    W = np.zeros((3, DIM), np.float32)
    W[0] = 50.0 * _DIRS[1]      # investigate <- the INVESTIGATE direction
    b = np.asarray([0.0, 1.0, -5.0], np.float32)
    small = _save_head("route_in", W, b, n=10,
                       per={"investigate": 5, "answer_directly": 5})
    h, why = e1.load("route_in")
    check(h is None and "MIN_TRAIN_N" in why,
          "a head trained on too few labels is not served", why)
    d, why = e1.decide("route_in", "anything at all")
    check(d is None and "MIN_TRAIN_N" in why,
          "decide returns None + why: the caller's rule decides", why)
    v = _save_head("route_in", W, b, n=60,
                   per={"investigate": 25, "answer_directly": 25,
                        "clarify": 10})
    h, why = e1.load("route_in")
    check(h is not None and h.version == v and why == "ok",
          "a promoted head with enough labels is served", why)
    blob = e1.artefact("route_in", v)
    check(blob["trained_on"]["n"] == 60 and blob["date"] and "cv" in blob
          and blob["seed"] == 0 and blob["labels"] == labels,
          "the artefact records n, CV, seed, date and labels")
    bad = dict(blob, W_b64=blob["W_b64"][:-8] + "AAAAAAA=")
    try:
        e1.Head(bad)
        ok = False
    except ValueError:
        ok = True
    check(ok, "corrupt weights are refused (sha256)")
    other = dict(blob, feature=dict(blob["feature"], embed_model="other"))
    check("contract" in (e1.Head(other).usable() or ""),
          "a head fitted on another embedding contract is not served")
    r = e1.revert("route_in")
    check(r["ok"] and r["current"] == small and e1.current_version(
        "route_in") == small, "revert serves the previous version", str(r))
    r = e1.revert("route_in", 0)
    check(r["ok"] and e1.load("route_in")[0] is None,
          "revert to 0 serves none: the rule decides")
    e1.promote("route_in", v, "back", author="test")
    check(e1.versions("route_in") == [small, v],
          "every version is kept on disk")


# ================================================================ 2 ========
def test_one_embedding_cached():
    _CALLS.clear()
    d1, s1 = e1.decide("route_in", "Where is INVESTIGATE defined in r185?")
    d2, s2 = e1.decide("route_in", "Where is INVESTIGATE defined in r185?")
    check(d1 is not None and d1["choice"] == "investigate" and s1 ==
          "answered", "the served head answers", f"{d1} {s1}")
    check(len(_CALLS) == 1 and d2["cached"] and not d1["cached"],
          "the same request is embedded once (cached per text)",
          f"{len(_CALLS)} calls")
    e1._MEM.clear()
    check(e1.cached_vector(e1.render("Where is INVESTIGATE defined in r185?"))
          is not None, "the vector is in the on-disk store too")
    check(d1["head_ms"] < 5.0, "the head's arithmetic is sub-millisecond",
          f"{d1['head_ms']} ms")
    real = e1._embed

    def dead(_t):
        raise ConnectionError("embedder down")
    e1._embed = dead
    try:
        d, why = e1.decide("route_in", "a brand new question never embedded")
    finally:
        e1._embed = real
    check(d is None and "embedder unavailable" in why,
          "a dead embedder is None + a reason, never a guess", why)


# ================================================================ 3 ========
def _select(question: str):
    import selection
    import tiers
    tiers._accepted = ("low", "medium", "xhigh")
    t = tiers.resolve({"reasoning_effort": "max"}, None)
    gate = {"offer": True, "situation": "REPOSITORY_BOUND"}
    return selection.select([{"role": "user", "content": question}], t, gate)


def test_the_proxy_never_calls_laya_with_e1_on():
    import fanout
    import selection
    import shomen
    import skill_select
    q = "Where is INVESTIGATE defined and what calls it?"
    _flag(False)
    _OPENED.clear()
    d = _select(q)
    check(any("/route" in u for u in _OPENED)
          and d["signals"].get("second_signal") == "Laya",
          "E1 off: Laya is still the second signal (and was asked)",
          f"{_OPENED} {d['signals'].get('laya_status')}")
    _flag(True)
    try:
        _OPENED.clear()
        d = _select(q)
        sig = d["signals"]
        check(sig.get("second_signal") == "E1"
              and (sig.get("laya_status") or "").startswith("answered (E1")
              and (sig.get("laya") or {}).get("engine") == "e1",
              "E1 on: selection's second signal is E1", json.dumps(
                  {k: sig.get(k) for k in ("second_signal", "laya_status")}))
        check("E1 agrees" in d["because"]["investigate"]
              or "E1 says" in d["because"]["investigate"],
              "the reason names E1, not Laya", d["because"]["investigate"])
        check(not _OPENED, "E1 on: no network call at all", str(_OPENED))
        check(selection.laya_signal("q")[0] is None
              and "YAMADORI_E1" in selection.laya_signal("q")[1],
              "selection.laya_signal sends nothing")
        check(skill_select.laya_pick("s", {"a": "x", "b": "y"}) is None,
              "skill_select.laya_pick sends nothing")
        try:
            shomen._laya("s", {})
            ok = False
        except RuntimeError as e:
            ok = "YAMADORI_E1" in str(e)
        check(ok, "shomen._laya refuses (its contract is to raise)")
        check(fanout._choice_averaged("s", "i", {"a": "x", "b": "y"}) is None,
              "fanout._choice_averaged sends nothing")
        check(not _OPENED, "and none of them opened a URL", str(_OPENED))
    finally:
        _flag(False)


# ================================================================ 4 ========
def _deep_decide(msgs, route_class="agent_step"):
    import deep
    import tiers
    tiers._accepted = ("low", "medium", "xhigh")
    t = tiers.resolve({"reasoning_effort": "xhigh"}, None)
    _deep_decide.n += 1
    return deep.decide(raw=msgs, tier=t, route={"class": route_class},
                       util={}, account="acct", lineage=f"l{_deep_decide.n}",
                       turn_key=f"k{_deep_decide.n}")


_deep_decide.n = 0


def test_phase06_consults_trained_heads_only():
    import deep
    fail = "npm ERR! Test failed.\nexit code 1"
    msgs = [{"role": "system", "content": "agent"},
            {"role": "user", "content": "Make the ESCALATE tests pass."},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "a", "type": "function", "function": {
                    "name": "terminal",
                    "arguments": json.dumps({"command": "npm test"})}}]},
            {"role": "tool", "tool_call_id": "a", "content": fail},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "b", "type": "function", "function": {
                    "name": "terminal",
                    "arguments": json.dumps({"command": "npm test"})}}]},
            {"role": "tool", "tool_call_id": "b", "content": fail}]
    n_ev = len(deep.struggle_events(msgs))
    check(1 <= n_ev < 3, "fixture: struggle below the rule's threshold",
          str(n_ev))
    _flag(False)
    r = _deep_decide(msgs)
    check(not r["fire"] and "e1" not in r["signals"] and "e1_state" not in r,
          "E1 off: the rule decides and nothing is embedded")
    _flag(True)
    try:
        r = _deep_decide(msgs)
        check(not r["fire"] and r["signals"]["e1"]["heads"].get("escalate")
              is None, "E1 on, escalate untrained: the rule still decides",
              r["because"])
        esc_dim = DIM + len(e1.HEADS["escalate"]["extra"])
        _save_head("escalate", np.zeros((2, esc_dim)), [8.0, -8.0], n=50,
                   per={"escalate": 25, "continue": 25})
        r = _deep_decide(msgs)
        check(r["fire"] and r["kind"] == "struggle"
              and "E1 escalate head" in r["because"],
              "E1 on, escalate trained: the head fires below the threshold "
              "and says so", r["because"])
        check(r.get("e1_state", "").startswith("question: Make the ESCALATE"),
              "the embedded state rides with the decision")
        check("e1_state" not in deep.public(r),
              "x_yamadori.deep does not carry the state text")
        rid = deep.record(account="acct", conversation="conv-e1",
                          tier="xhigh", route="agent_step", rec=r,
                          n_messages=len(msgs), ran=True, kind="struggle")
        con = deep._db()
        try:
            row = con.execute("SELECT e1_state FROM deep_decisions WHERE "
                              "id=?", (rid,)).fetchone()
        finally:
            con.close()
        check(row is not None and row[0] == r["e1_state"],
              "the state is stored with the row for the learner")
        check(all("e1_state" not in x for x in deep.rows(50)),
              "deep.rows never reports it")
        e1.revert("escalate", 0)
        q = [{"role": "user", "content": "What does INVESTIGATE default to "
                                         "in three r185?"}]
        r = _deep_decide(q, "library_question")
        check(r["fire"] and r["kind"] == "area"
              and "E1 route_in head" in r["because"],
              "a library question E1 says to investigate is a known-hard "
              "area", r["because"])
        r = _deep_decide(q, "prose")
        check(not r["fire"], "the same answer on another route class does "
                             "not fire", r["because"])
    finally:
        _flag(False)


# ================================================================ 5 ========
def _rows_for_learn(n: int, separable: bool, seed: int) -> None:
    import deep
    rnd = np.random.default_rng(seed)
    con = deep._db()
    try:
        for i in range(n):
            esc = bool(i % 2)
            if not separable:
                esc = bool(rnd.integers(0, 2))
            text = e1.render(f"learn row {seed}-{i}"
                             + (" ESCALATE" if (esc and separable) else ""))
            e1.put_vector(text, fake_embed([text])[0])
            sig = {"struggle": {"count": 2, "kinds": {"tool_error_repeat": 2}}}
            con.execute(
                "INSERT INTO deep_decisions(id,created,day,account,"
                "conversation,traffic,allowed,trigger,fired,ran,signals,"
                "label,labelled_at,e1_state) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,"
                "?,?)",
                (os.urandom(6).hex(), time.time(), "2026-09-24", "a", "c",
                 "client", 1, "none", 0, 0, json.dumps(sig),
                 "missed" if esc else "fine", time.time(), text))
    finally:
        con.close()


def test_self_tuning_promotes_only_on_a_paired_win():
    import dash_deep
    import deep_learn
    e1.revert("escalate", 0)
    before = set(e1.versions("escalate"))
    _rows_for_learn(80, separable=True, seed=1)
    _CALLS.clear()
    out = {r["head"]: r for r in e1.learn(author="test", seed=11)}
    esc = out["escalate"]
    check(esc.get("status") == "promoted"
          and esc["evaluation"]["baseline"] == "rule"
          and esc["evaluation"]["candidate_correct"]
          > esc["evaluation"]["baseline_correct"]
          and esc["evaluation"]["mcnemar"]["p"] < 0.05,
          "an untrained slot is promoted when it beats the rule, paired, "
          "p < 0.05", json.dumps(esc.get("evaluation"))[:300])
    check(not _CALLS, "learning embedded nothing (cpu lane)")
    v1 = e1.current_version("escalate")
    check(v1 not in before and e1.load("escalate")[0] is not None,
          "the promoted version is the one served")
    again = {r["head"]: r for r in e1.learn(author="test", seed=12)}
    check("skipped" in again["escalate"],
          "no new labels: nothing is retrained", str(again["escalate"]))
    _rows_for_learn(80, separable=False, seed=2)
    third = {r["head"]: r for r in e1.learn(author="test", seed=13)}
    t = third["escalate"]
    check(t.get("status") == "kept_old" and e1.current_version("escalate")
          == v1 and t["evaluation"]["baseline"] == f"v{v1}",
          "a candidate that does not win is recorded and the current head "
          "stays", json.dumps(t.get("evaluation"))[:300])
    check(e1.artefact("escalate", t["candidate"])["status"] == "kept_old",
          "the losing candidate is kept as a version, marked kept_old")
    ov = e1.overview()
    vs = ov["heads"]["escalate"]["versions"]
    check(all(x.get("n") for x in vs) and ov["heads"]["route_in"]["versions"],
          "the overview lists every version with its n")
    code, _ct, body = dash_deep.handle_get("/dash/api/deep")
    d = json.loads(body)
    check(code == 200 and "e1" in d and "escalate" in d["e1"]["heads"],
          "GET /dash/api/deep carries e1")
    check("learn row" not in body.decode(),
          "the dashboard payload carries no state text")
    code, _ct, body = dash_deep.handle_post("/dash/api/deep/e1/revert",
                                            {"head": "escalate"}, "op")
    check(code == 200 and e1.current_version("escalate") != v1,
          "POST /dash/api/deep/e1/revert serves an older version",
          body.decode()[:200])
    code, _ct, body = dash_deep.handle_post(
        "/dash/api/deep/e1/decide",
        {"head": "route_in", "question": "Where is INVESTIGATE set?"}, "op")
    dd = json.loads(body)
    check(code == 200 and dd["ok"] and dd["decision"]["choice"] ==
          "investigate" and "question" not in json.dumps(dd["decision"]),
          "POST /dash/api/deep/e1/decide answers with the served head",
          body.decode()[:200])
    res = deep_learn.handle_learn({"id": "j1"}, None)
    check("e1" in res or "deferred" in res,
          "the idle learner's job runs e1.learn", str(res)[:200])


def main() -> int:
    for fn in (test_the_contract, test_one_embedding_cached,
               test_the_proxy_never_calls_laya_with_e1_on,
               test_phase06_consults_trained_heads_only,
               test_self_tuning_promotes_only_on_a_paired_win):
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
    passed = sum(1 for ok, _, _ in _results if ok)
    total = len(_results)
    print(f"\n{'=' * 70}\n  {passed}/{total} checks passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
    try:
        sys.exit(main())
    finally:
        urllib.request.urlopen = _REAL_URLOPEN
