#!/usr/bin/env python
"""E1 -- the embedding-head classifier -- measured (docs/E1.md). Evaluation
and the one training run that ships route_in v1.

    PY=C:/Users/jwals/textgen/installer_files/env/python.exe
    $PY bench/e1/eval_e1.py embed        # single-text vectors -> E1 store
    $PY bench/e1/eval_e1.py stability    # fit stability per weight decay
    $PY bench/e1/eval_e1.py train        # route_in v1 (+ seeds, MLP check)
    $PY bench/e1/eval_e1.py set2         # E1 and the rule on both held-out sets
    $PY bench/e1/eval_e1.py latency      # head arithmetic + embedding call
    $PY bench/e1/eval_e1.py offline      # predictions the live check compares

HELD-OUT SETS. Set 1 is bench/laya_routing_heldout_packages.jsonl (120).
Set 2 (141) is labelled by hand against route_in-rubric-v1 before any arm
was run on it: labels in bench/e1/route_heldout2_labels.jsonl (ids, keys,
reasons), text in index/e1/heldout2_candidates.jsonl (gitignored -- the repo
is public and the text is from the corpus), joined by e1.heldout2_rows();
second blind pass on 29 rows: bench/e1/route_heldout2_second_pass.jsonl.
Nothing trains on either. No question text is printed or written here:
results carry ids, labels, predictions and counts.

LAYA, removed 2026-09-29 (the way back is commit e360d37): the Laya arms
(`laya`, `narrow`, the Laya and selection columns of `set2`, the torch
comparison in `stability`) went with it; their recorded numbers stay in
results/ and docs/E1.md.

THE CARD. Embeddings are on the A4000, shared with the Octopus run's
search. Before every call block this reads nvidia-smi by UUID and WAITS
while free memory is under 1,331 MiB, an image server is on the card, or
llama-swap does not report `embeddings` ready: it never loads or evicts
anything (a missing model is waited for, not loaded). Waits are recorded.

STATISTICS. Exact two-sided McNemar on the discordant pairs (e1.mcnemar),
exact Clopper-Pearson intervals (below; bench/eval_route_heldout.py's, moved
here when that script was removed with Laya).
"""
from __future__ import annotations

import argparse
import json
import os
import math
import subprocess
import sys
import time
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "mcp"))
sys.path.insert(0, os.path.join(ROOT, "bench"))

import e1  # noqa: E402

mcnemar_exact = e1.mcnemar

RESULTS = os.path.join(HERE, "results")
SET1 = os.path.join(ROOT, "bench", "laya_routing_heldout_packages.jsonl")
SET2 = "set2"            # e1.heldout2_rows(): labels in bench/, text in index/
# The batched vectors the Tev1 evaluation's control made (16 per call; it
# lived in bench/tev1/results/, removed 2026-09-29, moved here).
BATCHED = os.path.join(HERE, "results", "control_embeddings.npz")
STACK = os.environ.get("LLAMA_STACK_URL", "http://127.0.0.1:11434")
CARD_UUID = "GPU-43e37d0c-4104-9056-2552-6109d4d3382c"   # A4000 only
HEADROOM_MIB = 1331
LABELS = e1.HEADS["route_in"]["labels"]
WAITS: list[dict] = []


# ------------------------------------------------------------ stats ---------
def _binom_cdf(k: int, n: int, p: float) -> float:
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k + 1))


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact binomial CI by bisection on the binomial CDF (no scipy needed)."""
    if n == 0:
        return float("nan"), float("nan")

    def solve(f, lo=0.0, hi=1.0):
        for _ in range(100):
            mid = (lo + hi) / 2
            if f(mid):
                hi = mid
            else:
                lo = mid
        return (lo + hi) / 2

    lower = 0.0 if k == 0 else solve(lambda p: 1 - _binom_cdf(k - 1, n, p) >= alpha / 2)
    upper = 1.0 if k == n else solve(lambda p: _binom_cdf(k, n, p) <= alpha / 2)
    return lower, upper


# ------------------------------------------------------------ card ----------
def _a4000_free() -> int:
    out = subprocess.run(["nvidia-smi", f"--id={CARD_UUID}",
                          "--query-gpu=memory.free", "--format=csv,noheader,"
                          "nounits"], capture_output=True, text=True,
                         timeout=20).stdout.strip()
    return int(out)


def _image_server() -> bool:
    out = subprocess.run(["nvidia-smi", f"--id={CARD_UUID}",
                          "--query-compute-apps=process_name",
                          "--format=csv,noheader"], capture_output=True,
                         text=True, timeout=20).stdout
    return "sd-server" in out or "sd-cli" in out


def _embeddings_ready() -> bool:
    try:
        with urllib.request.urlopen(STACK + "/running", timeout=10) as r:
            d = json.load(r)
        return any(m.get("model") == "embeddings" and m.get("state") == "ready"
                   for m in d.get("running") or [])
    except Exception:                                            # noqa: BLE001
        return False


def wait_for_room(need_embeddings: bool = True, max_wait_s: float = 3600) -> None:
    t0 = time.time()
    while True:
        free = _a4000_free()
        ok = free >= HEADROOM_MIB and not _image_server() and \
            (not need_embeddings or _embeddings_ready())
        if ok:
            if time.time() - t0 > 1:
                WAITS.append({"waited_s": round(time.time() - t0, 1),
                              "free": free})
            return
        if time.time() - t0 > max_wait_s:
            raise SystemExit(f"no room on the A4000 for {max_wait_s} s "
                             f"(free {free} MiB): not run -- never evict")
        time.sleep(10)


# ------------------------------------------------------------ data ----------
def rows(path: str) -> list[dict]:
    if path == SET2:
        rs = e1.heldout2_rows()
        if not rs:
            raise SystemExit("held-out set 2 text is missing: run "
                             "bench/e1/build_heldout2.py (index/e1/)")
        return rs
    with open(path, encoding="utf-8") as fh:
        return [json.loads(x) for x in fh if x.strip()]


def texts(rs: list[dict]) -> list[str]:
    return [e1.render(r.get("question"), r.get("context")) for r in rs]


def X_of(rs: list[dict]) -> np.ndarray:
    out = []
    for t in texts(rs):
        v = e1.cached_vector(t)
        if v is None:
            raise SystemExit("a vector is missing from the E1 store: run "
                             "`eval_e1.py embed` first")
        out.append(v)
    return np.stack(out)


def save(name: str, blob) -> str:
    os.makedirs(RESULTS, exist_ok=True)
    p = os.path.join(RESULTS, name)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(blob, fh, indent=1, default=float)
    return p


def load(name: str):
    p = os.path.join(RESULTS, name)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def fmt(k: int, n: int) -> str:
    lo, hi = clopper_pearson(k, n)
    return f"{k}/{n} ({k / n:.3f}, 95% CI {lo:.3f}-{hi:.3f})"


def binary_ok(pred: list, truth: list[str]) -> list[bool]:
    return [(p == "investigate") == (t == "investigate")
            for p, t in zip(pred, truth)]


# ------------------------------------------------------------ embed ---------
def cmd_embed(args) -> None:
    """Every training, set-1 and set-2 state, embedded ONE TEXT PER CALL --
    exactly as E1 serves -- into the E1 store. Compared with the batched
    vectors (BATCHED, 16 per call) on the rows both have."""
    tr = e1.gold_route_rows()
    all_texts = [r["text"] for r in tr] + texts(rows(SET1)) + texts(rows(SET2))
    todo = [t for t in dict.fromkeys(all_texts) if e1.cached_vector(t) is None]
    print(f"{len(all_texts)} states, {len(todo)} to embed (one per call)")
    ms = []
    for i, t in enumerate(todo):
        if i % 25 == 0:
            wait_for_room()
        t0 = time.perf_counter()
        v = e1._embed([t])[0]
        ms.append((time.perf_counter() - t0) * 1000)
        if float(np.linalg.norm(v)) < 0.5:
            raise SystemExit("a non-unit embedding: the embedder is dead")
        e1.put_vector(t, v)
    z = np.load(BATCHED)
    batched = np.concatenate([z["tr"], z["te"]])
    single = np.stack([e1.cached_vector(t) for t in
                       [r["text"] for r in tr] + texts(rows(SET1))])
    cos = (batched * single).sum(1)
    blob = {"embedded": len(todo), "waits": WAITS,
            "embed_ms": _q(ms) if ms else None,
            "batched_vs_single_cosine": _q(cos.tolist()),
            "n_compared": int(len(cos))}
    print(json.dumps({k: v for k, v in blob.items()}, indent=1))
    print("->", save("embed.json", blob))


def _q(xs: list[float]) -> dict:
    s = sorted(xs)

    def q(p):
        return s[min(len(s) - 1, int(round(p * (len(s) - 1))))]
    return {"n": len(s), "min": round(s[0], 6), "p50": round(q(0.5), 6),
            "p95": round(q(0.95), 6), "max": round(s[-1], 6),
            "mean": round(sum(s) / len(s), 6)}


# ------------------------------------------------------------ stability -----
def cmd_stability(args) -> None:
    """Is the fit reproducible at each decay? float32 (the served fit)
    against the same algorithm in float64, on the batched vectors
    (BATCHED). The torch comparison (scripts/train_laya._fit_linear) went
    with Laya; its record is results/stability.json."""
    z = np.load(BATCHED)
    tr = e1.gold_route_rows()
    y = [LABELS.index(r["label"]) for r in tr]
    out = {}
    for wd in e1.WD_GRID_TRAIN_LAYA:
        W, b = e1.fit_logistic(z["tr"], y, 3, wd)
        W64, b64 = _fit64(z["tr"], y, 3, wd)
        p32 = e1.probs(W, b, z["te"]).argmax(1)
        p64 = (z["te"].astype(np.float64) @ W64.T + b64).argmax(1)
        rec = {"argmax_agree_f32_f64": int((p32 == p64).sum()), "n": 120}
        out[str(wd)] = rec
        print(wd, rec)
    print("->", save("stability.json", out))


def _fit64(X, y, k, wd, steps=e1.STEPS, lr=e1.LR):
    X = np.asarray(X, np.float64)
    n, d = X.shape
    mu, sd = X.mean(0), np.maximum(X.std(0, ddof=1), 1e-6)
    Xs = (X - mu) / sd
    Y = np.zeros((n, k))
    Y[np.arange(n), y] = 1
    W, b = np.zeros((k, d)), np.zeros(k)
    mW, vW, mb, vb = np.zeros_like(W), np.zeros_like(W), np.zeros(k), np.zeros(k)
    for t in range(1, steps + 1):
        zz = Xs @ W.T + b
        zz -= zz.max(1, keepdims=True)
        p = np.exp(zz)
        p /= p.sum(1, keepdims=True)
        G = (p - Y) / n
        gW, gb = G.T @ Xs + wd * W, G.sum(0) + wd * b
        mW, vW = .9 * mW + .1 * gW, .999 * vW + .001 * gW * gW
        mb, vb = .9 * mb + .1 * gb, .999 * vb + .001 * gb * gb
        c1, c2 = 1 - .9 ** t, 1 - .999 ** t
        W -= lr * (mW / c1) / (np.sqrt(vW / c2) + 1e-8)
        b -= lr * (mb / c1) / (np.sqrt(vb / c2) + 1e-8)
    return W / sd, b - (W * mu / sd).sum(1)


# ------------------------------------------------------------ train ---------
def cmd_train(args) -> None:
    """route_in v1: E1's recorded recipe (5-fold CV seed 0 over the stable
    decays, full fit) on SINGLE-text vectors, saved and promoted. Also: the
    batched-vector control reproduced, the 8 seeds, and an MLP check."""
    tr = e1.gold_route_rows()
    y = [LABELS.index(r["label"]) for r in tr]
    s1, s2 = rows(SET1), rows(SET2)
    t1 = [r["label"] for r in s1]
    t2 = [r["label"] for r in s2]
    Xtr, X1, X2 = X_of(tr), X_of(s1), X_of(s2)
    report: dict = {}

    # (a) the recorded control, on the batched vectors, full train_laya grid
    z = np.load(BATCHED)
    cvb = e1.cross_validate(z["tr"], y, 3, 0, grid=e1.WD_GRID_TRAIN_LAYA)
    Wb, bb = e1.fit_logistic(z["tr"], y, 3, cvb["wd"])
    pb = [LABELS[i] for i in e1.probs(Wb, bb, z["te"]).argmax(1)]
    report["control_batched_seed0"] = {"wd": cvb["wd"], "cv": cvb["accuracy"],
                                      "set1_binary": sum(binary_ok(pb, t1))}

    # (b) seeds 0..7, single-text vectors, both grids
    seeds = {}
    for grid_name, grid in (("stable", e1.WD_GRID),
                            ("train_laya", e1.WD_GRID_TRAIN_LAYA)):
        per = []
        for s in range(8):
            cv = e1.cross_validate(Xtr, y, 3, s, grid=grid)
            W, b = e1.fit_logistic(Xtr, y, 3, cv["wd"])
            p1 = [LABELS[i] for i in e1.probs(W, b, X1).argmax(1)]
            p2 = [LABELS[i] for i in e1.probs(W, b, X2).argmax(1)]
            ok1 = binary_ok(p1, t1)
            per.append({"seed": s, "wd": cv["wd"], "cv": cv["accuracy"],
                        "set1": sum(ok1), "set2": sum(binary_ok(p2, t2))})
        seeds[grid_name] = per
        print(grid_name, [(p["seed"], p["wd"], p["cv"], p["set1"], p["set2"])
                          for p in per])
    report["seeds"] = seeds

    # (c) MLP only if it beats logistic in CV (same folds, seed 0)
    report["mlp_check"] = _mlp_check(Xtr, y)
    print("mlp", report["mlp_check"])

    # (d) v1: seed 0, stable grid, single vectors -- the served head
    cv = e1.cross_validate(Xtr, y, 3, 0)
    W, b = e1.fit_logistic(Xtr, y, 3, cv["wd"])
    per_label = {lab: y.count(i) for i, lab in enumerate(LABELS)}
    srcs: dict = {}
    for r in tr:
        srcs[r["source"]] = srcs.get(r["source"], 0) + 1
    P1, P2 = e1.probs(W, b, X1), e1.probs(W, b, X2)
    p1 = [LABELS[i] for i in P1.argmax(1)]
    p2 = [LABELS[i] for i in P2.argmax(1)]
    ev = {"set1": {"n": len(s1), "binary": sum(binary_ok(p1, t1)),
                   "three_way": sum(p == t for p, t in zip(p1, t1))},
          "set2": {"n": len(s2), "binary": sum(binary_ok(p2, t2))},
          "script": "bench/e1/eval_e1.py train"}
    keys = [e1._row_key(dict(r)) for r in tr]
    existing = e1.current_version("route_in")
    if existing and not args.force:
        v = existing
        print(f"route_in v{v} already current; not re-saved (--force)")
    else:
        v = e1.save_version("route_in", W, b,
                            trained_on={"n": len(tr), "per_label": per_label,
                                        "sources": srcs},
                            cv={k: cv[k] for k in ("wd", "accuracy", "by_wd",
                                                   "folds", "seed")},
                            seed=0, author="bench/e1/eval_e1.py train",
                            evaluation=ev, keys=keys, parent=None,
                            status="promoted")
        e1.promote("route_in", v, "initial: E1's recorded recipe on the 289 "
                   "labels Laya's head used", author="bench/e1/eval_e1.py")
    report["v1"] = {"version": v, "wd": cv["wd"], "cv": cv["accuracy"],
                    "by_wd": cv["by_wd"], "oof_margin": _margins(cv["oof"]),
                    **ev}
    # predictions every other command reads (no text)
    preds = {}
    for name, P, rs in (("set1", P1, s1), ("set2", P2, s2)):
        order = np.argsort(-P, 1)
        preds[name] = {"choice": [LABELS[i] for i in order[:, 0]],
                       "top2": [[LABELS[order[j, 0]], LABELS[order[j, 1]]]
                                for j in range(len(rs))],
                       "margin": [float(P[j, order[j, 0]] - P[j, order[j, 1]])
                                  for j in range(len(rs))],
                       "probs": P.round(6).tolist(),
                       "ids": [r.get("id", i) for i, r in enumerate(rs)]}
    preds["oof_margin_train"] = [float(np.sort(p)[-1] - np.sort(p)[-2])
                                 for p in cv["oof"]]
    save("e1_preds.json", preds)
    print(json.dumps({k: v for k, v in report["v1"].items()}, indent=1))
    print("->", save("train.json", report))


def _margins(P) -> dict:
    m = [float(np.sort(p)[-1] - np.sort(p)[-2]) for p in P]
    return _q(m)


def _mlp_check(X: np.ndarray, y: list[int], hidden: int = 64, steps: int = 400,
               lr: float = 0.01) -> dict:
    """A one-hidden-layer MLP (64 ReLU, Adam, torch) against the logistic
    head on the SAME folds, 8 fold seeds, paired on the out-of-fold
    predictions (exact McNemar per seed). The MLP's weight decay is picked
    on seed 0 from three values -- a selection the logistic head does not
    get beyond its own two -- so a tie favours logistic. Adopted only if it
    is ahead with p < 0.05 on a majority of seeds. Never looks at a held-out
    set."""
    try:
        import torch
    except Exception as e:                                       # noqa: BLE001
        return {"not_run": f"{type(e).__name__}: {e}"}
    Y = np.asarray(y)

    def oof(seed: int, wd: float) -> np.ndarray:
        out = np.zeros(len(y), int)
        for a, bidx in e1.kfold(y, e1.CV_FOLDS, seed):
            torch.manual_seed(0)
            Xa = torch.tensor(X[a])
            mu, sd = Xa.mean(0), Xa.std(0).clamp_min(1e-6)
            net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], hidden),
                                      torch.nn.ReLU(),
                                      torch.nn.Linear(hidden, 3))
            opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
            ya = torch.tensor(Y[a])
            for _ in range(steps):
                opt.zero_grad()
                torch.nn.functional.cross_entropy(net((Xa - mu) / sd),
                                                  ya).backward()
                opt.step()
            with torch.no_grad():
                out[bidx] = net((torch.tensor(X[bidx]) - mu) / sd
                                ).argmax(1).numpy()
        return out
    by_wd = {wd: float((oof(0, wd) == Y).mean()) for wd in (1e-4, 1e-3, 1e-2)}
    wd = max(by_wd, key=lambda w: (by_wd[w], -w))
    seeds, wins = [], 0
    for s in range(8):
        lo = e1.cross_validate(X, y, 3, s)["oof"].argmax(1) == Y
        mo = oof(s, wd) == Y
        m = mcnemar_exact(lo.tolist(), mo.tolist())
        wins += int(mo.sum() > lo.sum() and m["p"] < 0.05)
        seeds.append({"seed": s, "logistic": int(lo.sum()),
                      "mlp": int(mo.sum()), "mcnemar": m})
    lm = sum(x["logistic"] for x in seeds) / (8 * len(y))
    mm = sum(x["mlp"] for x in seeds) / (8 * len(y))
    return {"mlp_wd_by_seed0": {str(k): round(v, 4) for k, v in by_wd.items()},
            "mlp_wd": wd, "seeds": seeds, "logistic_mean_oof": round(lm, 4),
            "mlp_mean_oof": round(mm, 4), "significant_mlp_wins": wins,
            "adopt_mlp": wins >= 5}


# ------------------------------------------------------------ set 2 ---------
def cmd_set2(args) -> None:
    """E1 and the rule on BOTH held-out sets, investigate-vs-not, paired.
    (The Laya-head and selection.decide columns were removed with Laya and
    selection's investigate decision, 2026-09-29; results/set2.json keeps
    their last numbers.)"""
    import selection
    out = {}
    for name, path in (("set1", SET1), ("set2", SET2)):
        rs = rows(path)
        truth = [r["label"] for r in rs]
        tb = [t == "investigate" for t in truth]
        e1c = load("e1_preds.json")[name]["choice"]
        rule = [selection.rule_baseline({"question": r["question"],
                                         "context": r.get("context") or ""})
                for r in rs]
        corr = {"E1": binary_ok(e1c, truth), "rule": binary_ok(rule, truth)}
        preds = {"E1": [c == "investigate" for c in e1c],
                 "rule": [c == "investigate" for c in rule]}
        res = {"n": len(rs), "investigate": sum(tb), "correct": {},
               "missed": {}, "unneeded": {}, "mcnemar": {}}
        for k, c in corr.items():
            res["correct"][k] = sum(c)
            res["missed"][k] = sum(1 for p, t in zip(preds[k], tb) if t and not p)
            res["unneeded"][k] = sum(1 for p, t in zip(preds[k], tb) if p and not t)
        res["mcnemar"]["rule vs E1"] = mcnemar_exact(corr["rule"], corr["E1"])
        if name == "set2":
            src = {}
            for i, r in enumerate(rs):
                s = r["source"]
                src.setdefault(s, {"n": 0, **{k: 0 for k in corr}})
                src[s]["n"] += 1
                for k in corr:
                    src[s][k] += corr[k][i]
            res["by_source"] = src
            res["three_way_e1"] = sum(p == t for p, t in zip(e1c, truth))
        out[name] = res
        print(f"\n{name}: n={len(rs)}, {sum(tb)} investigate")
        for k in corr:
            print(f"  {k:16s} {fmt(res['correct'][k], len(rs))}  missed "
                  f"{res['missed'][k]}  unneeded {res['unneeded'][k]}")
        for k, m in res["mcnemar"].items():
            print(f"  {k:32s} a-only {m['a_only']:3d} b-only {m['b_only']:3d} "
                  f"p={m['p']:.4g}")
        if name == "set2":
            for s, d in res["by_source"].items():
                print(f"  source {s}: {d}")
    print("->", save("set2_e1_rule.json", out))


# ------------------------------------------------------------ latency -------
def cmd_latency(args) -> None:
    """The head's arithmetic, timed (the embedding call is timed in
    embed.json, one text per call)."""
    h, why = e1.load("route_in")
    if h is None:
        raise SystemExit(f"no served route_in head: {why}")
    X = X_of(rows(SET1))
    ts = []
    for rep in range(20):
        for x in X:
            t0 = time.perf_counter()
            h.predict(x)
            ts.append((time.perf_counter() - t0) * 1000)
    blob = {"head_predict_ms": _q(ts), "calls": len(ts),
            "note": "Head.predict on one 1024-d vector: a 3x1024 affine map, "
                    "softmax, argsort, dict build; CPU, single thread"}
    print(json.dumps(blob, indent=1))
    print("->", save("latency.json", blob))


# ------------------------------------------------------------ offline -------
def cmd_offline(args) -> None:
    """What the live check compares against: the served head's choice and
    probabilities per held-out row (set 1 by index, set 2 by id), from
    single-text vectors -- the served condition."""
    h, why = e1.load("route_in")
    if h is None:
        raise SystemExit(f"no served route_in head: {why}")
    out = {"head": "route_in", "version": h.version, "n": h.n, "rows": []}
    for name, path in (("set1", SET1), ("set2", SET2)):
        rs = rows(path)
        X = X_of(rs)
        for i, (r, x) in enumerate(zip(rs, X)):
            p = h.predict(x)
            out["rows"].append({"set": name, "id": r.get("id", i),
                                "choice": p["choice"],
                                "probabilities": p["probabilities"]})
    print(f"{len(out['rows'])} rows, route_in v{h.version}")
    print("->", save("offline_preds.json", out))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["embed", "stability", "train", "set2",
                                    "latency", "offline"])
    ap.add_argument("--force", action="store_true",
                    help="train: save a new version even if one is current")
    a = ap.parse_args()
    globals()[f"cmd_{a.cmd}"](a)


if __name__ == "__main__":
    main()
