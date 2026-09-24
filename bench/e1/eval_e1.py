#!/usr/bin/env python
"""E1 -- the embedding-head classifier -- measured (docs/E1.md). Evaluation
and the one training run that ships route_in v1.

    PY=C:/Users/jwals/textgen/installer_files/env/python.exe
    $PY bench/e1/eval_e1.py embed        # single-text vectors -> E1 store
    $PY bench/e1/eval_e1.py stability    # fit stability per weight decay
    $PY bench/e1/eval_e1.py train        # route_in v1 (+ seeds, MLP check)
    $PY bench/e1/eval_e1.py laya         # Laya's answers on set 2 (live :1237)
    $PY bench/e1/eval_e1.py narrow       # E1 alone vs E1 -> Laya
    $PY bench/e1/eval_e1.py set2         # every arm on held-out set 2
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

THE CARD. Embeddings and Laya are on the A4000, shared with the Octopus run's
search. Before every call block this reads nvidia-smi by UUID and WAITS
while free memory is under 1,331 MiB, an image server is on the card, or
llama-swap does not report `embeddings` ready: it never loads or evicts
anything (a missing model is waited for, not loaded). Waits are recorded.

STATISTICS. Exact two-sided McNemar on the discordant pairs, exact
Clopper-Pearson intervals -- bench/eval_route_heldout.py's, imported.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "mcp"))
sys.path.insert(0, os.path.join(ROOT, "bench"))

import e1  # noqa: E402
from eval_route_heldout import clopper_pearson, mcnemar_exact  # noqa: E402

RESULTS = os.path.join(HERE, "results")
SET1 = os.path.join(ROOT, "bench", "laya_routing_heldout_packages.jsonl")
SET2 = "set2"            # e1.heldout2_rows(): labels in bench/, text in index/
BATCHED = os.path.join(ROOT, "bench", "tev1", "results", "control_embeddings.npz")
LAYA_STAGE = os.path.join(ROOT, "index", "laya_staging_20260922_191425")
LAYA_HEAD = os.path.join(ROOT, "index", "laya", "route_in.json")
LAYA_URL = os.environ.get("LAYA_URL", "http://127.0.0.1:1237")
STACK = os.environ.get("LLAMA_STACK_URL", "http://127.0.0.1:11434")
CARD_UUID = "GPU-43e37d0c-4104-9056-2552-6109d4d3382c"   # A4000 only
HEADROOM_MIB = 1331
LABELS = e1.HEADS["route_in"]["labels"]
WAITS: list[dict] = []


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
    vectors bench/tev1 made (16 per call) on the rows both have."""
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
    against the same algorithm in float64, on the batched vectors bench/tev1
    measured, and against torch (scripts/train_laya._fit_linear)."""
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
        try:
            sys.path.insert(0, os.path.join(ROOT, "scripts"))
            import train_laya as TL
            Wt, bt = TL._fit_linear(z["tr"].tolist(), y, 3, wd, seed=0)
            pt = np.asarray(TL.predict_probs(Wt, bt, z["te"].tolist()))
            rec["argmax_agree_numpy_torch"] = int((pt.argmax(1) == p32).sum())
            rec["max_prob_diff_numpy_torch"] = float(
                np.abs(pt - e1.probs(W, b, z["te"])).max())
        except Exception as e:                                   # noqa: BLE001
            rec["torch"] = f"not run: {type(e).__name__}: {e}"
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


# ------------------------------------------------------------ Laya ----------
def laya_set1() -> list[dict]:
    """Laya's served route_in head on set 1, from the cached frozen features
    (live == offline on 120/120, bench/laya_factcheck/live_route_check.txt)."""
    import importlib
    lh = importlib.import_module("laya_head")
    with open(os.path.join(LAYA_STAGE, "features_heldout.json"),
              encoding="utf-8") as fh:
        fe = json.load(fh)["by_state"]
    h = lh.TrainedHead.load(LAYA_HEAD)
    return [h.decide(fe[lh.render_state("route_in", r)]) for r in rows(SET1)]


def _post(url: str, body: dict, timeout: int = 60) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def laya_pair(state: str, pair: list[str]) -> dict:
    """Laya zero-shot `choice` over TWO route options (its own trained
    descriptions), averaged over both orders -- Laya is order-sensitive
    (shomen._choice_averaged's rule)."""
    import importlib
    spec = importlib.import_module("laya_head").TASKS["route_in"]["question"]
    acc = {k: 0.0 for k in pair}
    for order in (pair, pair[::-1]):
        d = _post(LAYA_URL + "/decide", {"state": state, "questions": {
            "pick": {"type": "choice", "instructions": spec["instructions"],
                     "criteria": {k: spec["criteria"][k] for k in order}}}})
        p = d["answers"]["pick"].get("probabilities") or {}
        for k in pair:
            acc[k] += float(p.get(k, 0.0)) / 2
    return {"choice": max(acc, key=acc.get), "probabilities": acc}


def cmd_laya(args) -> None:
    """Laya's answers that need the live service (read-only, :1237): the
    trained head on set 2 (/route engine trained), and the two-option
    zero-shot choices over E1's top two on both sets (/decide)."""
    import importlib
    lh = importlib.import_module("laya_head")
    e1p = load("e1_preds.json")
    if not e1p:
        raise SystemExit("run `eval_e1.py train` first")
    out = load("laya_live.json") or {"set2_trained": {}, "pair": {}}
    s2 = rows(SET2)
    for i, r in enumerate(s2):
        if r["id"] in out["set2_trained"]:
            continue
        if i % 20 == 0:
            wait_for_room(need_embeddings=False)
        d = _post(LAYA_URL + "/route", {"task": "route_in", "engine": "trained",
                                        "question": r["question"],
                                        "context": r.get("context") or ""})
        out["set2_trained"][r["id"]] = {k: d.get(k) for k in
                                        ("choice", "probabilities", "margin",
                                         "abstain", "elapsed_ms")}
    for name, rs, key in (("set1", rows(SET1), None), ("set2", s2, "id")):
        top2 = e1p[name]["top2"]
        for i, r in enumerate(rs):
            rid = f"{name}:{r[key] if key else i}"
            if rid in out["pair"]:
                continue
            if i % 20 == 0:
                wait_for_room(need_embeddings=False)
            out["pair"][rid] = laya_pair(lh.render_state("route_in", r),
                                         top2[i])
        save("laya_live.json", out)
    out["waits"] = WAITS
    print(f"set2 trained {len(out['set2_trained'])}, pairs {len(out['pair'])}")
    print("->", save("laya_live.json", out))


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
    laya1 = [d["choice"] for d in laya_set1()]
    laya1_ok = binary_ok(laya1, t1)
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
            m = mcnemar_exact(laya1_ok, ok1)
            per.append({"seed": s, "wd": cv["wd"], "cv": cv["accuracy"],
                        "set1": sum(ok1), "set2": sum(binary_ok(p2, t2)),
                        "vs_laya_set1": m})
        seeds[grid_name] = per
        print(grid_name, [(p["seed"], p["wd"], p["cv"], p["set1"], p["set2"],
                           round(p["vs_laya_set1"]["p"], 5)) for p in per])
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


# ------------------------------------------------------------ narrowing -----
ARMS = {
    "N0_e1": "E1 alone (argmax)",
    "N1_e1top2_layahead": "E1's top two -> Laya's trained head picks between "
                          "them (its probabilities restricted to the two)",
    "N2_e1top2_layazs": "E1's top two -> Laya zero-shot choice over the two "
                        "(its own option descriptions, both orders averaged)",
    "N3_lowmargin_layahead": "Laya's trained head (restricted to E1's top "
                             "two) only where E1's margin is under the 25th "
                             "percentile of its out-of-fold training margins",
    "N4_lowmargin_layazs": "as N3 with the zero-shot two-way choice",
    "N5_average": "exploratory: average of E1's and Laya's head "
                  "probabilities",
}


def _laya_head_set(name: str) -> list[dict]:
    if name == "set1":
        return laya_set1()
    live = load("laya_live.json") or {}
    return [live["set2_trained"][r["id"]] for r in rows(SET2)]


def narrow_arms(name: str, rs: list[dict]) -> dict:
    e1p = load("e1_preds.json")[name]
    cut = float(np.percentile(load("e1_preds.json")["oof_margin_train"], 25))
    laya = _laya_head_set(name)
    live = load("laya_live.json") or {"pair": {}}
    key = "id" if name == "set2" else None
    arms: dict[str, list] = {}
    arms["N0_e1"] = e1p["choice"]
    n1, n2, n3, n4, n5 = [], [], [], [], []
    for i, r in enumerate(rs):
        a, b = e1p["top2"][i]
        lp = laya[i]["probabilities"]
        head_pick = a if lp[a] >= lp[b] else b
        zs = live["pair"].get(f"{name}:{r[key] if key else i}")
        zs_pick = zs["choice"] if zs else None
        low = e1p["margin"][i] < cut
        n1.append(head_pick)
        n2.append(zs_pick)
        n3.append(head_pick if low else e1p["choice"][i])
        n4.append((zs_pick if low else e1p["choice"][i]))
        avg = {lab: (e1p["probs"][i][j] + lp[lab]) / 2
               for j, lab in enumerate(LABELS)}
        n5.append(max(avg, key=avg.get))
    arms.update(N1_e1top2_layahead=n1, N2_e1top2_layazs=n2,
                N3_lowmargin_layahead=n3, N4_lowmargin_layazs=n4,
                N5_average=n5)
    arms["laya_head_alone"] = [d["choice"] for d in laya]
    return {"arms": arms, "cut": cut,
            "low_margin_rows": sum(1 for m in e1p["margin"] if m < cut)}


def cmd_narrow(args) -> None:
    out = {"arms_preregistered": ARMS,
           "rule": "a combination is adopted only if it beats N0 with exact "
                   "McNemar p < 0.05 on set 1 AND is not behind N0 on set 2; "
                   "5 pre-registered combinations, so Bonferroni alpha is "
                   "0.01 -- reported beside the raw p"}
    for name, path in (("set1", SET1), ("set2", SET2)):
        rs = rows(path)
        truth = [r["label"] for r in rs]
        na = narrow_arms(name, rs)
        ok = {k: binary_ok(v, truth) for k, v in na["arms"].items()
              if all(p is not None for p in v)}
        res = {"n": len(rs), "cut": na["cut"],
               "low_margin_rows": na["low_margin_rows"], "correct": {},
               "vs_N0": {}}
        for k, v in ok.items():
            res["correct"][k] = sum(v)
            if k != "N0_e1":
                res["vs_N0"][k] = mcnemar_exact(ok["N0_e1"], v)
        out[name] = res
        print(f"\n{name} n={len(rs)} (low-margin cut {na['cut']:.3f}: "
              f"{na['low_margin_rows']} rows)")
        for k, c in res["correct"].items():
            m = res["vs_N0"].get(k)
            print(f"  {k:24s} {fmt(c, len(rs))}"
                  + (f"   vs N0: N0-only {m['a_only']} arm-only {m['b_only']} "
                     f"p={m['p']:.4g}" if m else ""))
    print("->", save("narrow.json", out))


# ------------------------------------------------------------ set 2 ---------
def _selection(rs: list[dict], second: list[dict | None], name: str
               ) -> list[bool]:
    tmp = tempfile.mkdtemp(prefix="e1_sel_")
    os.environ.setdefault("YAMADORI_CORPUS_DB", os.path.join(tmp, "c.sqlite3"))
    os.environ.setdefault("LLAMA_STACK_URL", "http://127.0.0.1:1")
    import domains
    import selection
    import tiers
    t = tiers.resolve({"reasoning_effort": "max"}, tiers.from_header(None))
    dbs = selection.symbol_dbs()
    out = []
    for r, s in zip(rs, second):
        msgs = ([{"role": "user", "content": r["context"]}]
                if r.get("context") else []) + \
            [{"role": "user", "content": r["question"]}]
        gate = domains.tool_admission(msgs, None)
        if s is None:
            d = selection.decide(msgs, t, gate, dbs=dbs)
        else:
            d = selection.decide(msgs, t, gate, laya=s,
                                 laya_status="answered", dbs=dbs,
                                 second=name)
        out.append(bool(d["investigate"]))
    return out


def cmd_set2(args) -> None:
    """Every arm on BOTH sets, investigate-vs-not, paired."""
    import selection
    out = {}
    for name, path in (("set1", SET1), ("set2", SET2)):
        rs = rows(path)
        truth = [r["label"] for r in rs]
        tb = [t == "investigate" for t in truth]
        e1c = load("e1_preds.json")[name]["choice"]
        laya = _laya_head_set(name)
        rule = [selection.rule_baseline({"question": r["question"],
                                         "context": r.get("context") or ""})
                for r in rs]
        sel = _selection(rs, [None] * len(rs), "Laya")
        sel_l = _selection(rs, [{"choice": d["choice"], "abstain":
                                 bool(d.get("abstain")), "margin":
                                 d.get("margin")} for d in laya], "Laya")
        sel_e = _selection(rs, [{"choice": c, "abstain": False, "margin": None}
                                for c in e1c], "E1")
        corr = {"laya_head": binary_ok([d["choice"] for d in laya], truth),
                "E1": binary_ok(e1c, truth),
                "rule": binary_ok(rule, truth),
                "selection": [p == t for p, t in zip(sel, tb)],
                "selection+laya": [p == t for p, t in zip(sel_l, tb)],
                "selection+E1": [p == t for p, t in zip(sel_e, tb)]}
        res = {"n": len(rs), "investigate": sum(tb), "correct": {},
               "missed": {}, "unneeded": {}, "mcnemar": {}}
        preds = {"laya_head": [d["choice"] == "investigate" for d in laya],
                 "E1": [c == "investigate" for c in e1c],
                 "rule": [c == "investigate" for c in rule],
                 "selection": sel, "selection+laya": sel_l,
                 "selection+E1": sel_e}
        for k, c in corr.items():
            res["correct"][k] = sum(c)
            res["missed"][k] = sum(1 for p, t in zip(preds[k], tb) if t and not p)
            res["unneeded"][k] = sum(1 for p, t in zip(preds[k], tb) if p and not t)
        for a, b in (("laya_head", "E1"), ("rule", "E1"), ("selection", "E1"),
                     ("selection+laya", "selection+E1"),
                     ("selection+E1", "E1"), ("laya_head", "rule"),
                     ("selection", "laya_head")):
            res["mcnemar"][f"{a} vs {b}"] = mcnemar_exact(corr[a], corr[b])
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
            res["laya_abstained"] = sum(1 for d in laya if d.get("abstain"))
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
    print("->", save("set2.json", out))


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
    ap.add_argument("cmd", choices=["embed", "stability", "train", "laya",
                                    "narrow", "set2", "latency", "offline"])
    ap.add_argument("--force", action="store_true",
                    help="train: save a new version even if one is current")
    a = ap.parse_args()
    globals()[f"cmd_{a.cmd}"](a)


if __name__ == "__main__":
    main()
