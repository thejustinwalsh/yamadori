#!/usr/bin/env python
"""E1: the embedding-head classifier -- Yamadori's decision layer (docs/E1.md).

WHAT IT IS. A registry of small heads, one per decision, each a multinomial
logistic regression over the Qwen3-Embedding-0.6B vector of the REQUEST,
computed once by the resident `embeddings` model (code_search.embed, so the
gpu_room lease applies) and cached per text. A head is an affine map: its
inference is arithmetic over one vector (docs/E1.md measures it). It replaces
Laya's trained route_in head, which scored 80/120 on the held-out package
questions against E1's 104/120 (docs/TEV1-EVAL.md; re-measured in docs/E1.md
with this module's own trainer).

    route_in       investigate / answer_directly / clarify. Trained on the 289
                   labels Laya's head used (bench/laya_routing_labels*.jsonl).
    escalate       struggle: escalate now, or continue. UNTRAINED: labelled
                   from deep_decisions outcomes as they accumulate.
    kickoff        a new task: plan it first, or act. UNTRAINED, same source.
    skill_applies  does this armed skill apply to the request? UNTRAINED:
                   labelled from skill_learn's fallback records
                   (index/skills/router_labels.jsonl).

A head with fewer than MIN_TRAIN_N labels (or under MIN_PER_LABEL of a label)
is never served: `decide` returns None and the caller falls back to its rule.
MIN_TRAIN_N and MIN_PER_LABEL are CHOICES, not measurements.

ONE EMBEDDING PER REQUEST. Every head reads the same vector: the embedding of
`render(question, context)` -- the route_in state, byte for byte what Laya's
trainer rendered (laya_head.render_state), with the retrieval query
instruction code_search.embed prepends. Heads that need more than the text
append their own small scalar features (the struggle counts, the spec size);
skill_applies reads the product of the request and skill vectors. The
feature contract (embedding model, instruction, render revision, extras) is
stored in each artefact and checked before serving: a head fitted on one
rendering and served another would silently degrade.

ARTEFACTS. index/e1/heads/<head>/v0001.json ... -- weights (float32,
base64), the training n and per-label counts, the CV accuracy and the weight
decay it chose, the fold seed, the date, the evaluation that promoted or
rejected it. index/e1/heads/<head>/state.json names the CURRENT version and
keeps a log of every promotion and revert. Every version is kept; revert()
serves an older one. index/ is not tracked by git, as with index/laya.

SELF-TUNING (learn(), called by deep_learn's idle job on the cpu lane): when
a head has at least LEARN_MIN_NEW labels its current version never saw, a
candidate is fitted on the current version's rows plus 70% of the new ones
and scored against the current version (or, for an untrained slot, the
rule) on the other 30% -- rows NEITHER was trained on -- with an exact
McNemar test. It is promoted only when it is ahead AND p < PROMOTE_P;
otherwise it is recorded and the current version stays. route_in also has a
gold guard: the candidate may not be significantly worse on the two
hand-labelled held-out sets. The learner embeds nothing (cpu lane): a row
whose vector is not in the store is counted and skipped, and vectors reach
the store when E1 serves a request (YAMADORI_E1=1).

THE FLAG. YAMADORI_E1=1 turns E1 on in the proxy: selection's second signal
and deep.py's trigger consults read E1, and nothing on the request path calls
Laya (laya_allowed()). Default OFF until the live check in
mcp/test_live_stack.py (`e1`) passes.
"""
from __future__ import annotations

import base64
import datetime as dt
import glob
import hashlib
import json
import math
import os
import random
import sqlite3
import threading
import time
from collections import OrderedDict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
BENCH = os.path.join(ROOT, "bench")
E1_DIR = os.environ.get("YAMADORI_E1_DIR", os.path.join(ROOT, "index", "e1"))

# ------------------------------------------------------------ choices -------
# None of these is measured; each is a choice, named so it can be moved.
MIN_TRAIN_N = int(os.environ.get("YAMADORI_E1_MIN_TRAIN_N", "40"))
MIN_PER_LABEL = int(os.environ.get("YAMADORI_E1_MIN_PER_LABEL", "8"))
LEARN_MIN_NEW = int(os.environ.get("YAMADORI_E1_LEARN_MIN_NEW", "30"))
MIN_EVAL_N = int(os.environ.get("YAMADORI_E1_MIN_EVAL_N", "20"))
EVAL_FRAC = 0.3
PROMOTE_P = 0.05
# The fit is scripts/train_laya.py's _fit_linear, re-implemented in numpy so
# the worker needs no torch: standardise, zero init, full-batch Adam (lr
# 0.05, weight decay on every parameter, as torch applies it), 600 steps,
# folded back into one affine map. Weight decay by 5-fold CV over WD_GRID.
# The grid is train_laya's (1e-3 .. 1.0) cut to the two decays whose fit is
# numerically STABLE: at 0.1 and 1.0 Adam at lr 0.05 never settles (the
# L2 pull and the normalised step fight near zero), and the final iterate is
# float noise -- float32 vs float64 give the same held-out argmax on only
# 89/120 and 109/120 rows, against 120/120 at 1e-3 and 1e-2
# (bench/e1/eval_e1.py stability; docs/E1.md). The served v1 is 1e-2.
WD_GRID = (1e-3, 1e-2)
WD_GRID_TRAIN_LAYA = (1e-3, 1e-2, 1e-1, 1.0)
CV_FOLDS = 5
STEPS = 600
LR = 0.05
RENDER_REV = 1
STATE_CHARS = 6000
CACHE_SIZE = 512

HEADS: dict[str, dict] = {
    "route_in": {
        "labels": ["investigate", "answer_directly", "clarify"],
        "extra": [],
        "rule": "selection.rule_baseline (+ the held-symbol lookup)",
        "what": "does answering need source read first?",
    },
    "escalate": {
        "labels": ["escalate", "continue"],
        "extra": ["struggle_count", "tool_error_repeat", "file_rewritten",
                  "failing_command_rerun", "fixup_capped",
                  "user_still_broken"],
        "rule": "deep.py: struggle signals >= struggle_threshold",
        "what": "is the agent stuck enough to escalate now?",
    },
    "kickoff": {
        "labels": ["plan", "act"],
        "extra": ["log_tokens"],
        "rule": "deep.py: a new task whose spec is >= kickoff_tokens",
        "what": "should this new task be planned by the second brain?",
    },
    "skill_applies": {
        "labels": ["applies", "not_applies"],
        "extra": [],
        "pair": True,
        "rule": "skill_select stages 1-3 (and the model fallback)",
        "what": "does this armed skill apply to the request?",
    },
}


def enabled() -> bool:
    """YAMADORI_E1=1: E1 decides in the proxy and Laya is off the request
    path. Read per call so a test can flip it."""
    return os.environ.get("YAMADORI_E1", "0").strip() == "1"


def laya_allowed() -> bool:
    """False when E1 is on: every Laya caller on the request path checks
    this first (selection.laya_signal, skill_select.laya_pick,
    shomen._laya, fanout._choice_averaged)."""
    return not enabled()


# ------------------------------------------------------------ render --------
def render(question: str, context: str = "") -> str:
    """The route_in state -- laya_head.render_state("route_in", ...) byte for
    byte (mcp/test_e1.py asserts it), capped at STATE_CHARS."""
    q = (question or "").strip()
    ctx = (context or "").strip()
    s = f"question: {q}\ncontext: {ctx}" if ctx else f"question: {q}"
    return s[:STATE_CHARS]


def _contract() -> dict:
    """What a vector means: the embedding model, the query instruction and
    the render revision. Stored in every artefact and checked on load."""
    import code_search as cs
    return {"embed_model": cs.EMBED_MODEL,
            "instruct_sha": hashlib.sha256(
                cs.QUERY_INSTRUCT.encode()).hexdigest()[:16],
            "render_rev": RENDER_REV, "is_query": True}


def _key(text: str) -> str:
    c = _contract()
    return hashlib.sha256((c["embed_model"] + "\x00" + c["instruct_sha"]
                           + "\x00" + text).encode()).hexdigest()


# ------------------------------------------------------------ vectors -------
_MEM: "OrderedDict[str, np.ndarray]" = OrderedDict()
_MEM_LOCK = threading.Lock()
_STORE_OK: set[str] = set()


def _store_path() -> str:
    return os.path.join(E1_DIR, "vectors.sqlite3")


def _store() -> sqlite3.Connection:
    path = _store_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    con = sqlite3.connect(path, timeout=30, isolation_level=None)
    con.execute("PRAGMA busy_timeout=30000")
    if path not in _STORE_OK:
        con.execute("CREATE TABLE IF NOT EXISTS vec(key TEXT PRIMARY KEY, "
                    "dim INTEGER NOT NULL, vec BLOB NOT NULL, "
                    "created REAL NOT NULL)")
        _STORE_OK.add(path)
    return con


def cached_vector(text: str) -> np.ndarray | None:
    """The vector of `text` if it was ever embedded here; never embeds."""
    k = _key(text)
    with _MEM_LOCK:
        v = _MEM.get(k)
        if v is not None:
            _MEM.move_to_end(k)
            return v
    try:
        con = _store()
        try:
            row = con.execute("SELECT dim, vec FROM vec WHERE key=?",
                              (k,)).fetchone()
        finally:
            con.close()
    except sqlite3.Error:
        row = None
    if row is None:
        return None
    v = np.frombuffer(row[1], dtype=np.float32).copy()
    _remember(k, v)
    return v


def _remember(k: str, v: np.ndarray) -> None:
    with _MEM_LOCK:
        _MEM[k] = v
        _MEM.move_to_end(k)
        while len(_MEM) > CACHE_SIZE:
            _MEM.popitem(last=False)


def put_vector(text: str, v: np.ndarray) -> None:
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    k = _key(text)
    _remember(k, v)
    try:
        con = _store()
        try:
            con.execute("INSERT OR REPLACE INTO vec(key, dim, vec, created) "
                        "VALUES(?,?,?,?)", (k, int(v.shape[0]), v.tobytes(),
                                            time.time()))
        finally:
            con.close()
    except sqlite3.Error:
        pass


def _embed(texts: list[str]) -> np.ndarray:
    """A seam: code_search.embed with the query instruction (gpu_room lease
    inside). Replaced in tests."""
    import code_search as cs
    return cs.embed(texts, is_query=True)


def vector(text: str) -> tuple[np.ndarray, dict]:
    """(unit vector, {cached, embed_ms}). One embedding call on a miss."""
    v = cached_vector(text)
    if v is not None:
        return v, {"cached": True, "embed_ms": 0.0}
    t0 = time.perf_counter()
    v = np.asarray(_embed([text])[0], dtype=np.float32)
    ms = (time.perf_counter() - t0) * 1000
    if not np.isfinite(v).all() or float(np.linalg.norm(v)) < 0.5:
        # PROTOCOL rule 1: a dead embedder must not look like a live one.
        raise RuntimeError("the embedder returned a non-unit vector")
    put_vector(text, v)
    return v, {"cached": False, "embed_ms": round(ms, 2)}


# ------------------------------------------------------------ features ------
def extras_of(head: str, extra: dict | None) -> list[float]:
    names = HEADS[head]["extra"]
    e = extra or {}
    return [float(e.get(n) or 0.0) for n in names]


def features(head: str, v: np.ndarray, extra: dict | None = None,
             skill_vec: np.ndarray | None = None) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    if HEADS[head].get("pair"):
        if skill_vec is None:
            raise ValueError("skill_applies needs the skill's vector")
        s = np.asarray(skill_vec, dtype=np.float32).reshape(-1)
        return np.concatenate([v * s, [float(v @ s)]]).astype(np.float32)
    ex = extras_of(head, extra)
    return (np.concatenate([v, np.asarray(ex, dtype=np.float32)])
            if ex else v)


# ------------------------------------------------------------ fitting -------
def fit_logistic(X: np.ndarray, y: list[int], k: int, wd: float,
                 steps: int = STEPS, lr: float = LR
                 ) -> tuple[np.ndarray, np.ndarray]:
    """scripts/train_laya.py `_fit_linear` in numpy float32: standardise
    (unbiased std, as torch), zero init, full-batch Adam with L2 weight decay
    on weight AND bias, cross-entropy; folded back into (W, b) so serving
    needs no stored statistics. Zero init + full batch: no randomness."""
    X = np.asarray(X, dtype=np.float32)
    n, d = X.shape
    mu = X.mean(0)
    sd = np.maximum(X.std(0, ddof=1) if n > 1 else np.ones(d, np.float32),
                    1e-6).astype(np.float32)
    Xs = (X - mu) / sd
    Y = np.zeros((n, k), dtype=np.float32)
    Y[np.arange(n), np.asarray(y)] = 1.0
    W = np.zeros((k, d), dtype=np.float32)
    b = np.zeros(k, dtype=np.float32)
    mW, vW = np.zeros_like(W), np.zeros_like(W)
    mb, vb = np.zeros_like(b), np.zeros_like(b)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(1, steps + 1):
        z = Xs @ W.T + b
        z = z - z.max(1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(1, keepdims=True)
        G = (p - Y) / n
        gW = G.T @ Xs + wd * W
        gb = G.sum(0) + wd * b
        mW = b1 * mW + (1 - b1) * gW
        vW = b2 * vW + (1 - b2) * gW * gW
        mb = b1 * mb + (1 - b1) * gb
        vb = b2 * vb + (1 - b2) * gb * gb
        c1, c2 = 1 - b1 ** t, 1 - b2 ** t
        W = W - lr * (mW / c1) / (np.sqrt(vW / c2) + eps)
        b = b - lr * (mb / c1) / (np.sqrt(vb / c2) + eps)
    W_eff = (W / sd).astype(np.float32)
    b_eff = (b - (W * mu / sd).sum(1)).astype(np.float32)
    return W_eff, b_eff


def probs(W: np.ndarray, b: np.ndarray, X: np.ndarray) -> np.ndarray:
    z = np.atleast_2d(np.asarray(X, dtype=np.float32)) @ W.T + b
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def kfold(y: list[int], k: int, seed: int) -> list[tuple[list[int], list[int]]]:
    """scripts/train_laya.py `kfold` with no groups, reproduced exactly (the
    string sort of indices included) so its seeds reproduce."""
    rnd = random.Random(seed)
    by_class: dict[int, list[int]] = {}
    for i in range(len(y)):
        by_class.setdefault(y[i], []).append(i)
    folds: list[list[int]] = [[] for _ in range(k)]
    for _c, gids in sorted(by_class.items(), key=lambda kv: str(kv[0])):
        gids = sorted(gids, key=str)
        rnd.shuffle(gids)
        for j, gid in enumerate(gids):
            folds[j % k].append(gid)
    out = []
    for j in range(k):
        val = folds[j]
        tr = [i for jj in range(k) if jj != j for i in folds[jj]]
        if val and tr:
            out.append((tr, val))
    return out


def cross_validate(X: np.ndarray, y: list[int], k: int, seed: int = 0,
                   grid=WD_GRID) -> dict:
    """{wd, accuracy, by_wd, folds, seed, oof}: weight decay chosen by
    CV_FOLDS-fold CV inside the rows given, ties to the larger decay (as
    bench/tev1's control did). `oof` holds the out-of-fold probabilities at
    the chosen decay."""
    folds = kfold(y, CV_FOLDS, seed)
    by_wd, oof_by = {}, {}
    for wd in grid:
        accs = []
        oof = np.zeros((len(y), k), dtype=np.float32)
        for a, bb in folds:
            W, b = fit_logistic(X[a], [y[i] for i in a], k, wd)
            p = probs(W, b, X[bb])
            oof[bb] = p
            accs.append(float((p.argmax(1) == np.asarray(y)[bb]).mean()))
        by_wd[wd] = sum(accs) / len(accs)
        oof_by[wd] = oof
    wd = max(by_wd, key=lambda w: (by_wd[w], -w))
    return {"wd": wd, "accuracy": round(by_wd[wd], 4),
            "by_wd": {str(w): round(a, 4) for w, a in by_wd.items()},
            "folds": CV_FOLDS, "seed": seed, "oof": oof_by[wd]}


def mcnemar(a_ok: list[bool], b_ok: list[bool]) -> dict:
    """Exact two-sided McNemar: a_only = a right & b wrong."""
    a = sum(1 for x, y in zip(a_ok, b_ok) if x and not y)
    b = sum(1 for x, y in zip(a_ok, b_ok) if y and not x)
    n = a + b
    if n == 0:
        return {"a_only": a, "b_only": b, "p": 1.0}
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(min(a, b) + 1)) / 2 ** n)
    return {"a_only": a, "b_only": b, "p": p}


# ------------------------------------------------------------ artefacts -----
def _head_dir(head: str) -> str:
    return os.path.join(E1_DIR, "heads", head)


def _state_path(head: str) -> str:
    return os.path.join(_head_dir(head), "state.json")


def _read_json(path: str, default):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return default


def _write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1)
    os.replace(tmp, path)


def versions(head: str) -> list[int]:
    out = []
    for p in glob.glob(os.path.join(_head_dir(head), "v*.json")):
        try:
            out.append(int(os.path.basename(p)[1:-5]))
        except ValueError:
            pass
    return sorted(out)


def artefact(head: str, version: int) -> dict | None:
    return _read_json(os.path.join(_head_dir(head), f"v{version:04d}.json"),
                      None)


def save_version(head: str, W: np.ndarray, b: np.ndarray, *, trained_on: dict,
                 cv: dict, seed: int, author: str, evaluation: dict | None,
                 keys: list[str], parent: int | None,
                 status: str) -> int:
    """Write a new version (never overwrites one). Returns its number."""
    v = (versions(head) or [0])[-1] + 1
    W = np.asarray(W, dtype=np.float32)
    raw = W.tobytes()
    blob = {
        "artefact": "e1-head", "format": 1, "head": head, "version": v,
        "labels": HEADS[head]["labels"], "kind": "logistic",
        "feature": dict(_contract(), extra=HEADS[head]["extra"],
                        pair=bool(HEADS[head].get("pair")),
                        dim=int(W.shape[1])),
        "shape": list(W.shape),
        "W_b64": base64.b64encode(raw).decode(),
        "W_sha256": hashlib.sha256(raw).hexdigest(),
        "b": [float(x) for x in np.asarray(b, dtype=np.float32)],
        "trained_on": trained_on,
        "trained_keys": keys,
        "fit": {"kind": "logistic", "wd": cv.get("wd"), "steps": STEPS,
                "lr": LR, "standardised": True, "init": "zeros"},
        "cv": {k: cv[k] for k in ("accuracy", "by_wd", "folds", "seed", "wd")
               if k in cv},
        "seed": seed, "date": dt.date.today().isoformat(),
        "created": time.time(), "author": author, "parent": parent,
        "evaluation": evaluation or {}, "status": status,
    }
    _write_json(os.path.join(_head_dir(head), f"v{v:04d}.json"), blob)
    return v


def current_version(head: str) -> int | None:
    st = _read_json(_state_path(head), {})
    return st.get("current")


def _log_state(head: str, current: int | None, event: dict) -> None:
    st = _read_json(_state_path(head), {"current": None, "log": []})
    st["current"] = current
    st.setdefault("log", []).append(dict(event, at=time.time(),
                                         current=current))
    _write_json(_state_path(head), st)
    _HEAD_CACHE.pop(head, None)


def promote(head: str, version: int, why: str, author: str = "learner") -> dict:
    if artefact(head, version) is None:
        return {"ok": False, "error": f"no {head} v{version}"}
    old = current_version(head)
    _log_state(head, version, {"event": "promote", "from": old,
                               "to": version, "why": why, "author": author})
    return {"ok": True, "head": head, "current": version, "previous": old}


def revert(head: str, version: int | None = None,
           author: str = "operator") -> dict:
    """Serve an older version (default: the one before the current), or
    version 0 to serve none (the caller's rule decides). Recorded."""
    if head not in HEADS:
        return {"ok": False, "error": f"unknown head {head!r}"}
    cur = current_version(head)
    if version is None:
        older = [v for v in versions(head) if cur is None or v < cur]
        version = older[-1] if older else 0
    if version and artefact(head, version) is None:
        return {"ok": False, "error": f"no {head} v{version}"}
    _log_state(head, version or None, {"event": "revert", "from": cur,
                                       "to": version or None,
                                       "author": author})
    return {"ok": True, "head": head, "current": version or None,
            "previous": cur}


class Head:
    def __init__(self, blob: dict):
        self.blob = blob
        self.head = blob["head"]
        self.version = int(blob["version"])
        self.labels = list(blob["labels"])
        raw = base64.b64decode(blob["W_b64"])
        if hashlib.sha256(raw).hexdigest() != blob["W_sha256"]:
            raise ValueError(f"{self.head} v{self.version}: weights corrupt")
        self.W = np.frombuffer(raw, dtype=np.float32).reshape(blob["shape"])
        self.b = np.asarray(blob["b"], dtype=np.float32)
        self.n = int((blob.get("trained_on") or {}).get("n") or 0)
        self.per_label = (blob.get("trained_on") or {}).get("per_label") or {}

    def usable(self) -> str | None:
        """None when servable; otherwise why not."""
        if self.n < MIN_TRAIN_N:
            return f"trained on {self.n} labels (< MIN_TRAIN_N {MIN_TRAIN_N})"
        thin = [lab for lab in self.labels
                if int(self.per_label.get(lab, 0)) < MIN_PER_LABEL]
        if len(self.labels) - len(thin) < 2:
            return f"fewer than two labels with >= {MIN_PER_LABEL} examples"
        want = dict(_contract(), extra=HEADS[self.head]["extra"],
                    pair=bool(HEADS[self.head].get("pair")))
        have = {k: self.blob["feature"].get(k) for k in want}
        if have != want:
            return f"feature contract differs: artefact {have}, served {want}"
        return None

    def predict(self, x: np.ndarray) -> dict:
        p = probs(self.W, self.b, x)[0]
        order = np.argsort(-p)
        return {"choice": self.labels[int(order[0])],
                "probabilities": {lab: round(float(p[i]), 6)
                                  for i, lab in enumerate(self.labels)},
                "margin": round(float(p[order[0]] - p[order[1]]), 6),
                "abstain": False, "engine": "e1", "head": self.head,
                "version": self.version, "n": self.n}


_HEAD_CACHE: dict[str, tuple[float, Head | None, str | None]] = {}


def load(head: str) -> tuple[Head | None, str]:
    """(the current servable head, status). Re-read when state.json
    changes."""
    sp = _state_path(head)
    try:
        mt = os.path.getmtime(sp)
    except OSError:
        return None, "untrained: no version has been promoted"
    hit = _HEAD_CACHE.get(head)
    if hit and hit[0] == mt:
        return hit[1], hit[2] or "ok"
    cur = current_version(head)
    h, why = None, None
    if not cur:
        why = "no current version (reverted to the rule)"
    else:
        blob = artefact(head, cur)
        if blob is None:
            why = f"current v{cur} is missing"
        else:
            try:
                h = Head(blob)
                why = h.usable()
                if why:
                    h = None
            except Exception as e:                               # noqa: BLE001
                h, why = None, f"v{cur} failed to load: {e}"
    _HEAD_CACHE[head] = (mt, h, why)
    return h, why or "ok"


# ------------------------------------------------------------ serving -------
def decide(head: str, question: str, context: str = "",
           extra: dict | None = None, skill_text: str | None = None
           ) -> tuple[dict | None, str]:
    """(prediction, status) for one request; (None, why) when the head is
    untrained, too small, or the embedder did not answer -- the caller's
    rule then decides. Never raises."""
    h, why = load(head)
    if h is None:
        return None, why
    try:
        text = render(question, context)
        v, info = vector(text)
        sv = None
        if HEADS[head].get("pair"):
            sv, _i = vector(render(skill_text or ""))
        t0 = time.perf_counter()
        out = h.predict(features(head, v, extra, sv))
        out["head_ms"] = round((time.perf_counter() - t0) * 1000, 4)
        out.update(info)
        return out, "answered"
    except Exception as e:                                       # noqa: BLE001
        return None, f"embedder unavailable ({type(e).__name__}: {str(e)[:160]})"


def consult(question: str, context: str = "",
            extras: dict | None = None) -> dict:
    """Every non-pair head on ONE embedding of the request, for deep.py:
    {state, heads: {name: prediction | None}, status: {name: why}, embed}.
    A head that is not servable reports why; the embedding is made only if
    at least one head is."""
    text = render(question, context)
    out: dict = {"state": text, "heads": {}, "status": {}, "embed": None}
    live = {}
    for name, spec in HEADS.items():
        if spec.get("pair"):
            continue
        h, why = load(name)
        if h is None:
            out["heads"][name], out["status"][name] = None, why
        else:
            live[name] = h
    if not live:
        return out
    try:
        v, info = vector(text)
        out["embed"] = info
    except Exception as e:                                       # noqa: BLE001
        for name in live:
            out["heads"][name] = None
            out["status"][name] = (f"embedder unavailable ({type(e).__name__}"
                                   f": {str(e)[:120]})")
        return out
    for name, h in live.items():
        t0 = time.perf_counter()
        p = h.predict(features(name, v, (extras or {}).get(name)))
        p["head_ms"] = round((time.perf_counter() - t0) * 1000, 4)
        out["heads"][name], out["status"][name] = p, "answered"
    return out


def route_signal(question: str, context: str = "") -> tuple[dict | None, str]:
    """selection's second signal, shaped like laya_signal's reply."""
    d, status = decide("route_in", question, context)
    if d is None:
        return None, f"E1 route_in: {status}"
    return d, f"answered (E1 route_in v{d['version']}, n={d['n']})"


# ------------------------------------------------------------ labels --------
def gold_route_rows() -> list[dict]:
    """The 289 route_in training labels, as scripts/train_laya.load_labels
    reads them: every bench/laya_routing_labels*.jsonl, sorted by path,
    de-duplicated on the rendered state."""
    rows, seen = [], set()
    for path in sorted(glob.glob(os.path.join(BENCH,
                                              "laya_routing_labels*.jsonl"))):
        with open(path, encoding="utf-8") as fh:
            for ln in fh:
                if not ln.strip():
                    continue
                r = json.loads(ln)
                text = render(r.get("question"), r.get("context"))
                if text in seen:
                    continue
                seen.add(text)
                rows.append({"text": text, "label": r["label"],
                             "source": os.path.basename(path),
                             "question": r.get("question"),
                             "context": r.get("context") or ""})
    return rows


SET1_PATH = os.path.join(BENCH, "laya_routing_heldout_packages.jsonl")
# Held-out set 2: the LABELS (ids, keys, reasons -- no text) are in bench/,
# the request TEXT is in index/e1/ (gitignored: the repo is public and the
# text is from the corpus). bench/e1/build_heldout2.py writes the text.
SET2_LABELS = os.path.join(BENCH, "e1", "route_heldout2_labels.jsonl")
SET2_TEXT = os.path.join(E1_DIR, "heldout2_candidates.jsonl")


def _qkey(question: str, context: str) -> str:
    return hashlib.sha256(((question or "") + "\x00" + (context or ""))
                          .encode()).hexdigest()[:12]


def heldout2_rows() -> list[dict]:
    """Set 2 joined: labels from bench/, text from index/e1/, on id, with
    the key (sha256 of question NUL context) checked. [] when the text file
    is absent (a fresh clone); raises on a key mismatch -- a rebuilt
    candidate file must not be silently re-labelled."""
    if not (os.path.exists(SET2_LABELS) and os.path.exists(SET2_TEXT)):
        return []
    with open(SET2_TEXT, encoding="utf-8") as fh:
        text = {r["id"]: r for r in (json.loads(x) for x in fh if x.strip())}
    out = []
    with open(SET2_LABELS, encoding="utf-8") as fh:
        for ln in fh:
            if not ln.strip():
                continue
            lab = json.loads(ln)
            t = text.get(lab["id"])
            if t is None or _qkey(t.get("question"), t.get("context")) \
                    != lab["key"]:
                raise ValueError(f"held-out set 2: {lab['id']} does not match "
                                 f"its text (rebuilt candidates?)")
            out.append(dict(t, label=lab["label"], why=lab.get("why"),
                            rubric=lab.get("rubric")))
    return out


def gold_heldout_rows() -> list[dict]:
    rows1 = []
    if os.path.exists(SET1_PATH):
        with open(SET1_PATH, encoding="utf-8") as fh:
            rows1 = [json.loads(x) for x in fh if x.strip()]
    out = []
    for src, rs in (("set1", rows1), ("set2", heldout2_rows())):
        for r in rs:
            out.append({"text": render(r.get("question"), r.get("context")),
                        "label": r["label"], "question": r.get("question"),
                        "context": r.get("context") or "", "source": src})
    return out


# deep_decisions outcome -> the label each head learns from it. A label that
# says nothing about the decision (skipped, unobserved, not_helped: the run
# happened and the struggle went on -- neither "should not have run" nor
# "helped") is not used. CHOICES, stated in docs/E1.md.
def _deep_label(head: str, r: dict) -> str | None:
    lab, ran, trig = r.get("label"), bool(r.get("ran")), r.get("trigger")
    try:
        sig = json.loads(r.get("signals") or "{}")
    except ValueError:
        sig = {}
    if head == "escalate":
        if int((sig.get("struggle") or {}).get("count") or 0) < 1:
            return None
        if ran and trig == "struggle":
            return {"helped": "escalate", "wasted": "continue"}.get(lab)
        if not ran:
            return {"missed": "escalate", "escalated_later": "escalate",
                    "fine": "continue"}.get(lab)
        return None
    if head == "kickoff":
        if not (sig.get("kickoff") or {}).get("new_task"):
            return None
        if ran and trig == "kickoff":
            return {"helped": "plan", "wasted": "act"}.get(lab)
        if not ran:
            return {"missed": "plan", "escalated_later": "plan",
                    "fine": "act"}.get(lab)
        return None
    if head == "route_in":
        if r.get("route") not in ("library_question", "prose"):
            return None
        if ran:
            return {"helped": "investigate",
                    "wasted": "answer_directly"}.get(lab)
        return {"missed": "investigate", "escalated_later": "investigate",
                "fine": "answer_directly"}.get(lab)
    return None


def _deep_extra(head: str, r: dict) -> dict:
    try:
        sig = json.loads(r.get("signals") or "{}")
    except ValueError:
        sig = {}
    if head == "escalate":
        st = sig.get("struggle") or {}
        kinds = st.get("kinds") or {}
        return dict({"struggle_count": st.get("count") or 0}, **kinds)
    if head == "kickoff":
        ko = sig.get("kickoff") or {}
        return {"log_tokens": math.log1p(float(ko.get("tokens") or 0))}
    return {}


def _deep_rule(head: str, r: dict) -> str:
    trig = r.get("trigger")
    if head == "escalate":
        return "escalate" if trig == "struggle" else "continue"
    if head == "kickoff":
        return "plan" if trig == "kickoff" else "act"
    return "investigate" if r.get("fired") else "answer_directly"


def live_rows(head: str) -> list[dict]:
    """Labelled client rows of deep_decisions for `head`, with the state
    text E1 embedded when it served the request (deep.py stores it)."""
    if head == "skill_applies":
        return _skill_rows()
    try:
        import deep
        con = deep._db()
    except Exception:                                            # noqa: BLE001
        return []
    try:
        cols = {c[1] for c in con.execute("PRAGMA table_info(deep_decisions)")}
        if "e1_state" not in cols:
            return []
        rs = [dict(x) for x in con.execute(
            "SELECT * FROM deep_decisions WHERE traffic='client' AND label "
            "IS NOT NULL AND e1_state IS NOT NULL ORDER BY created")]
    finally:
        con.close()
    out = []
    for r in rs:
        lab = _deep_label(head, r)
        if lab is None:
            continue
        out.append({"text": r["e1_state"], "label": lab,
                    "extra": _deep_extra(head, r), "source": "deep_decisions",
                    "rule": _deep_rule(head, r), "id": r["id"]})
    return out


def _skill_rows() -> list[dict]:
    try:
        import skill_learn
        path = skill_learn.LABELS
    except Exception:                                            # noqa: BLE001
        return []
    if not os.path.exists(path):
        return []
    texts = _skill_texts()
    out = []
    with open(path, encoding="utf-8") as fh:
        for ln in fh:
            try:
                r = json.loads(ln)
            except ValueError:
                continue
            st = texts.get(r.get("skill"))
            if not st:
                continue
            out.append({"text": render(r.get("query") or ""),
                        "skill_text": st,
                        "label": "applies" if r.get("label") else "not_applies",
                        "source": "skill_fallbacks", "rule": "not_applies",
                        "id": f"{r.get('fallback')}:{r.get('skill')}"})
    return out


def skill_text(skill: dict) -> str:
    """What a skill is embedded as: its title and its applies-when text."""
    return (f"{skill.get('title') or ''}: "
            f"{(skill.get('rule') or {}).get('text') or ''}").strip()


def _skill_texts() -> dict[str, str]:
    try:
        import skills as store
        return {s["id"]: skill_text(s) for s in store.armed()}
    except Exception:                                            # noqa: BLE001
        return {}


def _row_key(r: dict) -> str:
    return hashlib.sha256((r["text"] + "\x00" + (r.get("skill_text") or "")
                           + "\x00" + r["label"]).encode()).hexdigest()[:20]


def _row_features(head: str, r: dict) -> np.ndarray | None:
    v = cached_vector(r["text"])
    if v is None:
        return None
    sv = None
    if HEADS[head].get("pair"):
        sv = cached_vector(render(r.get("skill_text") or ""))
        if sv is None:
            return None
    return features(head, v, r.get("extra"), sv)


def rows_for(head: str) -> list[dict]:
    rows = (gold_route_rows() if head == "route_in" else []) + live_rows(head)
    for r in rows:
        r["key"] = _row_key(r)
    return rows


def _rule_choice(head: str, r: dict) -> str:
    if r.get("rule"):
        return r["rule"]
    if head == "route_in":
        import selection
        return selection.rule_baseline({"question": r.get("question") or "",
                                        "context": r.get("context") or ""})
    return HEADS[head]["labels"][-1]


def _decision_ok(head: str, pred: str, truth: str) -> bool:
    """What the decision uses: route_in is scored investigate-vs-not (the
    binary selection and deep.py read); the others are two-way already."""
    if head == "route_in":
        return (pred == "investigate") == (truth == "investigate")
    return pred == truth


def train(head: str, rows: list[dict], seed: int = 0) -> dict:
    """Fit one candidate on `rows` (which must carry cached vectors):
    {W, b, cv, trained_on, keys} or {error}."""
    labels = HEADS[head]["labels"]
    X, y, keys, per, srcs = [], [], [], {}, {}
    for r in rows:
        x = _row_features(head, r)
        if x is None:
            continue
        X.append(x)
        y.append(labels.index(r["label"]))
        keys.append(r["key"])
        per[r["label"]] = per.get(r["label"], 0) + 1
        srcs[r.get("source") or "?"] = srcs.get(r.get("source") or "?", 0) + 1
    n = len(y)
    if n < MIN_TRAIN_N:
        return {"error": f"{n} labelled rows with vectors (< MIN_TRAIN_N "
                         f"{MIN_TRAIN_N})", "n": n}
    if sum(1 for lab in labels if per.get(lab, 0) >= MIN_PER_LABEL) < 2:
        return {"error": f"fewer than two labels with >= {MIN_PER_LABEL} "
                         f"examples: {per}", "n": n}
    Xa = np.stack(X)
    cv = cross_validate(Xa, y, len(labels), seed)
    W, b = fit_logistic(Xa, y, len(labels), cv["wd"])
    return {"W": W, "b": b, "cv": cv, "keys": keys,
            "trained_on": {"n": n, "per_label": per, "sources": srcs}}


def learn(author: str = "learner", seed: int | None = None) -> list[dict]:
    """One self-tuning pass over every head (deep_learn calls it). Returns
    one record per head: skipped (why), or a candidate version with its
    paired test and whether it was promoted. Embeds nothing."""
    out = []
    for head in HEADS:
        try:
            out.append(_learn_head(head, author, seed))
        except Exception as e:                                   # noqa: BLE001
            out.append({"head": head, "skipped": f"raised {type(e).__name__}:"
                                                 f" {e}"[:300]})
    return out


def _learn_head(head: str, author: str, seed: int | None) -> dict:
    rows = rows_for(head)
    cur_v = current_version(head)
    cur_blob = artefact(head, cur_v) if cur_v else None
    seen = set((cur_blob or {}).get("trained_keys") or [])
    with_vec = [r for r in rows if _row_features(head, r) is not None]
    no_vec = len(rows) - len(with_vec)
    new = [r for r in with_vec if r["key"] not in seen]
    rec: dict = {"head": head, "labelled": len(rows), "no_vector": no_vec,
                 "new": len(new), "current": cur_v}
    if len(new) < LEARN_MIN_NEW:
        rec["skipped"] = (f"{len(new)} new labelled rows with vectors "
                          f"(< LEARN_MIN_NEW {LEARN_MIN_NEW})")
        return rec
    s = int(seed if seed is not None else int(time.time()) % 100000)
    rnd = random.Random(s)
    by: dict[str, list[dict]] = {}
    for r in sorted(new, key=lambda r: r["key"]):
        by.setdefault(r["label"], []).append(r)
    test, rest = [], []
    for lab in sorted(by):
        g = by[lab][:]
        rnd.shuffle(g)
        cut = int(round(len(g) * EVAL_FRAC))
        test += g[:cut]
        rest += g[cut:]
    if len(test) < MIN_EVAL_N:
        rec["skipped"] = (f"a held-out split of {len(test)} rows (< "
                          f"MIN_EVAL_N {MIN_EVAL_N}); waiting for labels")
        return rec
    train_rows = [r for r in with_vec if r["key"] in seen] + rest
    cand = train(head, train_rows, seed=s % 8)
    if "error" in cand:
        rec["skipped"] = cand["error"]
        return rec
    labels = HEADS[head]["labels"]
    Xt = np.stack([_row_features(head, r) for r in test])
    cand_pred = [labels[i] for i in probs(cand["W"], cand["b"], Xt).argmax(1)]
    cur_head = Head(cur_blob) if cur_blob else None
    if cur_head is not None:
        base_pred = [labels[i] for i in
                     probs(cur_head.W, cur_head.b, Xt).argmax(1)]
        base_name = f"v{cur_v}"
    else:
        base_pred = [_rule_choice(head, r) for r in test]
        base_name = "rule"
    c_ok = [_decision_ok(head, p, r["label"]) for p, r in zip(cand_pred, test)]
    b_ok = [_decision_ok(head, p, r["label"]) for p, r in zip(base_pred, test)]
    m = mcnemar(b_ok, c_ok)
    ev = {"split_seed": s, "n_test": len(test), "baseline": base_name,
          "candidate_correct": sum(c_ok), "baseline_correct": sum(b_ok),
          "mcnemar": m, "promote_p": PROMOTE_P}
    promote_it = sum(c_ok) > sum(b_ok) and m["p"] < PROMOTE_P
    if promote_it and head == "route_in" and cur_head is not None:
        g = _gold_guard(cur_head, cand)
        ev["gold_guard"] = g
        if g.get("worse"):
            promote_it = False
    status = "promoted" if promote_it else "kept_old"
    v = save_version(head, cand["W"], cand["b"],
                     trained_on=cand["trained_on"], cv=cand["cv"], seed=s,
                     author=author, evaluation=ev, keys=cand["keys"],
                     parent=cur_v, status=status)
    if promote_it:
        promote(head, v, f"beat {base_name} on {len(test)} held-out rows: "
                         f"{sum(c_ok)} vs {sum(b_ok)}, p={m['p']:.3g}",
                author)
    rec.update(candidate=v, status=status, evaluation=ev,
               n=cand["trained_on"]["n"])
    return rec


def _gold_guard(cur: Head, cand: dict) -> dict:
    """route_in only: the candidate must not be significantly worse than the
    current head on the hand-labelled held-out sets (vectors cached by
    bench/e1/eval_e1.py). Rows without a vector are skipped and counted."""
    rows = gold_heldout_rows()
    X, truth = [], []
    for r in rows:
        v = cached_vector(r["text"])
        if v is not None:
            X.append(v)
            truth.append(r["label"])
    if not X:
        return {"n": 0, "skipped": "no cached held-out vectors"}
    labels = HEADS["route_in"]["labels"]
    Xa = np.stack(X)
    a = [labels[i] for i in probs(cur.W, cur.b, Xa).argmax(1)]
    c = [labels[i] for i in probs(cand["W"], cand["b"], Xa).argmax(1)]
    a_ok = [_decision_ok("route_in", p, t) for p, t in zip(a, truth)]
    c_ok = [_decision_ok("route_in", p, t) for p, t in zip(c, truth)]
    m = mcnemar(a_ok, c_ok)
    return {"n": len(truth), "current": sum(a_ok), "candidate": sum(c_ok),
            "mcnemar": m,
            "worse": sum(c_ok) < sum(a_ok) and m["p"] < PROMOTE_P}


# ------------------------------------------------------------ overview ------
def overview() -> dict:
    """For GET /dash/api/deep: every head, its current version and every
    version with n, CV accuracy, status and evaluation. No text."""
    heads = {}
    for name, spec in HEADS.items():
        h, why = load(name)
        vs = []
        for v in versions(name):
            a = artefact(name, v) or {}
            vs.append({"version": v, "date": a.get("date"),
                       "n": (a.get("trained_on") or {}).get("n"),
                       "per_label": (a.get("trained_on") or {}).get("per_label"),
                       "sources": (a.get("trained_on") or {}).get("sources"),
                       "cv_accuracy": (a.get("cv") or {}).get("accuracy"),
                       "wd": (a.get("fit") or {}).get("wd"),
                       "seed": a.get("seed"), "status": a.get("status"),
                       "author": a.get("author"),
                       "evaluation": a.get("evaluation")})
        st = _read_json(_state_path(name), {})
        heads[name] = {"labels": spec["labels"], "what": spec["what"],
                       "rule": spec["rule"], "current": current_version(name),
                       "served": h is not None, "status": why,
                       "versions": vs, "log": (st.get("log") or [])[-20:]}
    return {"enabled": enabled(), "min_train_n": MIN_TRAIN_N,
            "min_per_label": MIN_PER_LABEL, "learn_min_new": LEARN_MIN_NEW,
            "min_eval_n": MIN_EVAL_N, "promote_p": PROMOTE_P,
            "heads": heads,
            "note": "MIN_TRAIN_N, MIN_PER_LABEL, LEARN_MIN_NEW, MIN_EVAL_N and "
                    "the outcome-to-label mapping are choices, unmeasured; "
                    "route_in's numbers are in docs/E1.md"}


if __name__ == "__main__":
    print(json.dumps(overview(), indent=1, default=str))
