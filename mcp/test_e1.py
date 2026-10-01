#!/usr/bin/env python
"""E1, the embedding-head classifier (mcp/e1.py). No GPU, no network.

    python mcp/test_e1.py      -> "N/M checks passed"

WHAT THIS IS GATING (docs/E1.md):

  1. THE CONTRACT: render() is laya_head.render_state("route_in") byte for
     byte (pinned below: laya_head and scripts/train_laya were removed with
     Laya, 2026-09-29); the numpy fit is deterministic; a head artefact
     round-trips (weights hashed), refuses to serve below MIN_TRAIN_N or
     with a different feature contract, and every version is kept,
     promotable and revertible.
  2. ONE EMBEDDING: a request is embedded once and cached (memory and the
     store); a dead embedder is None + a reason, never a guess.
  3. NO LAYA: with YAMADORI_E1=1 route_signal is E1's and the last Laya
     caller (skill_select.laya_pick) sends nothing; consult() reads every
     served head on one embedding.
  4. SELF-TUNING: e1.learn promotes a candidate only on a paired win on rows
     neither model trained on, keeps (and records) a loser, embeds nothing,
     and the overview shows every version with its n.

REMOVED 2026-09-29 (the way back is commit e360d37): the checks of
selection's Laya signal, shomen's and fanout's Laya calls, deep.py's E1
consults (the escalate head) and dash_deep / deep_learn -- their modules
are gone.

Heads here are synthetic (random unit vectors, 1024-d) in a temporary
E1_DIR; the real route_in v1 and its numbers are bench/e1/eval_e1.py's.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
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
os.environ.pop("YAMADORI_E1", None)

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
WORDS = ("ANSWER", "INVESTIGATE", "CLARIFY")


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
    # laya_head.render_state("route_in", rec), as it was at e360d37: the
    # rendering the 289 route_in labels were embedded under.
    for rec, want in (
            ({"question": "  Where is X defined? ", "context": ""},
             "question: Where is X defined?"),
            ({"question": "q", "context": " we use three 0.185 "},
             "question: q\ncontext: we use three 0.185"),
            ({"question": "", "context": None}, "question: ")):
        check(e1.render(rec.get("question"), rec.get("context")) == want,
              f"render is the route_in rendering for {rec!r}",
              repr(e1.render(rec.get("question"), rec.get("context"))))
    y = [0, 1, 2, 0, 1, 2, 0, 0, 1, 2, 1, 0, 2, 2, 1, 0, 0, 1]
    folds = e1.kfold(y, 5, 0)
    check(folds == e1.kfold(y, 5, 0)
          and sorted(i for _tr, va in folds for i in va) == list(range(len(y))),
          "kfold is deterministic and every row is validated once")
    X =_rng.normal(size=(60, 16)).astype(np.float32)
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
def test_no_laya_with_e1_on():
    q = "Where is INVESTIGATE defined and what calls it?"
    _flag(True)
    try:
        _OPENED.clear()
        d, status = e1.route_signal(q)
        check(d is not None and d["engine"] == "e1"
              and d["choice"] == "investigate"
              and status.startswith("answered (E1 route_in"),
              "route_signal answers with E1's route_in head", status)
        check(not e1.laya_allowed(), "E1 on: laya_allowed() is False")
        import skill_select
        if hasattr(skill_select, "laya_pick"):
            check(skill_select.laya_pick("s", {"a": "x", "b": "y"}) is None,
                  "skill_select.laya_pick sends nothing")
        out = e1.consult(q)
        check(set(out["heads"]) == {n for n, s in e1.HEADS.items()
                                    if not s.get("pair")}
              and out["heads"]["route_in"]["choice"] == "investigate"
              and out["embed"] is not None,
              "consult reads every served non-pair head on one embedding",
              json.dumps(out["status"]))
        check(not _OPENED, "E1 on: no network call at all", str(_OPENED))
    finally:
        _flag(False)
    check(e1.laya_allowed(), "E1 off: laya_allowed() is True")
    check("escalate" not in e1.HEADS,
          "the escalate head went with deep.py (2026-09-29)")


# ================================================================ 4 ========
_LEARN_ROWS: list[dict] = []


def _rows_for_learn(n: int, separable: bool, seed: int) -> None:
    """Synthetic route_in rows through the live_rows seam (deep_decisions,
    their old source, went with deep.py). The rule they are scored against
    says answer_directly every time."""
    rnd = np.random.default_rng(seed)
    for i in range(n):
        inv = bool(i % 2)
        if not separable:
            inv = bool(rnd.integers(0, 2))
        text = e1.render(f"learn row {seed}-{i}"
                         + (" INVESTIGATE" if (inv and separable) else ""))
        e1.put_vector(text, fake_embed([text])[0])
        _LEARN_ROWS.append({"text": text,
                            "label": "investigate" if inv else
                            "answer_directly",
                            "source": "test", "rule": "answer_directly",
                            "id": f"{seed}-{i}"})


def test_self_tuning_promotes_only_on_a_paired_win():
    real = e1.live_rows
    e1.live_rows = lambda head: ([dict(r) for r in _LEARN_ROWS]
                                 if head == "route_in" else [])
    try:
        e1.revert("route_in", 0)
        before = set(e1.versions("route_in"))
        _rows_for_learn(80, separable=True, seed=1)
        _CALLS.clear()
        out = {r["head"]: r for r in e1.learn(author="test", seed=11)}
        ri = out["route_in"]
        check(ri.get("status") == "promoted"
              and ri["evaluation"]["baseline"] == "rule"
              and ri["evaluation"]["candidate_correct"]
              > ri["evaluation"]["baseline_correct"]
              and ri["evaluation"]["mcnemar"]["p"] < 0.05,
              "an unserved slot is promoted when it beats the rule, paired, "
              "p < 0.05", json.dumps(ri.get("evaluation"))[:300])
        check(not _CALLS, "learning embedded nothing (cpu lane)")
        v1 = e1.current_version("route_in")
        check(v1 not in before and e1.load("route_in")[0] is not None,
              "the promoted version is the one served")
        again = {r["head"]: r for r in e1.learn(author="test", seed=12)}
        check("skipped" in again["route_in"],
              "no new labels: nothing is retrained", str(again["route_in"]))
        _rows_for_learn(80, separable=False, seed=2)
        third = {r["head"]: r for r in e1.learn(author="test", seed=13)}
        t = third["route_in"]
        check(t.get("status") == "kept_old" and e1.current_version("route_in")
              == v1 and t["evaluation"]["baseline"] == f"v{v1}",
              "a candidate that does not win is recorded and the current "
              "head stays", json.dumps(t.get("evaluation"))[:300])
        check(e1.artefact("route_in", t["candidate"])["status"] == "kept_old",
              "the losing candidate is kept as a version, marked kept_old")
        ov = e1.overview()
        vs = ov["heads"]["route_in"]["versions"]
        check(all(x.get("n") for x in vs) and "learn row" not in
              json.dumps(ov), "the overview lists every version with its n "
              "and carries no state text")
    finally:
        e1.live_rows = real


def main() -> int:
    for fn in (test_the_contract, test_one_embedding_cached,
               test_no_laya_with_e1_on,
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
    try:
        sys.exit(main())
    finally:
        urllib.request.urlopen = _REAL_URLOPEN
