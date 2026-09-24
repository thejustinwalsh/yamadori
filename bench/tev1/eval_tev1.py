#!/usr/bin/env python
"""Evaluate Tev1-4B-experimental as Yamadori's decision classifier (a
candidate to replace Laya). Evaluation only: nothing here is wired into the
stack, and nothing trains on a held-out file.

    PY=C:/Users/jwals/textgen/installer_files/env/python.exe
    $PY bench/tev1/eval_tev1.py sanity   --quant Q4_K_M
    $PY bench/tev1/eval_tev1.py latency  --quant Q4_K_M
    $PY bench/tev1/eval_tev1.py heldout  --quant Q4_K_M
    $PY bench/tev1/eval_tev1.py route    --quant Q4_K_M
    $PY bench/tev1/eval_tev1.py phase06  --quant Q4_K_M
    $PY bench/tev1/eval_tev1.py control           # E1: embeddings head
    $PY bench/tev1/eval_tev1.py report            # tables, both quants

THE SERVER is our own llama-server on --url (default :10091), started by
hand on the A4000 only (CUDA_VISIBLE_DEVICES = the A4000's UUID), with the
official PrismML build (config.yaml macro `server`). See docs/TEV1-EVAL.md.

GPU ROOM. The A4000 is shared with retrieval, Laya and image generation.
Before every block of requests this reads nvidia-smi by UUID and WAITS while
free memory is below HEADROOM_MIB (1,331, gpu_room.HEADROOM_MIB); it never
unloads anything. Waits are recorded in the result file.

PAIRING (PROTOCOL rule 6). Every arm runs on the same rows; comparisons are
exact two-sided McNemar on the discordant pairs, with n printed. Stats come
from bench/eval_route_heldout.py (the one copy).

THE ANSWER. The recommended request (no grammar) is scored as sent: the
content must be exactly one listed letter or the row counts as WRONG and as
INVALID (reported separately, PROTOCOL rule 3). The best listed letter from
the first token's top-20 logprobs (what the card's per-request regex would
pick) is recorded beside it as `constrained`.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
from collections import Counter
from typing import Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "bench"))
sys.path.insert(0, os.path.join(ROOT, "mcp"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import adapters as A  # noqa: E402
from eval_route_heldout import clopper_pearson, mcnemar_exact  # noqa: E402

RESULTS = os.path.join(HERE, "results")
CARD_UUID = "GPU-43e37d0c-4104-9056-2552-6109d4d3382c"   # A4000; never the 5060 Ti
HEADROOM_MIB = 1331
HELD_OUT = os.path.join(ROOT, "bench", "laya_routing_heldout_packages.jsonl")
HELD_ROWS = os.path.join(ROOT, "bench", "laya_factcheck", "heldout_rows.jsonl")
V0 = r"C:\Users\jwals\octo\logs\pilot-V0-xhigh-1"

WAITS: List[Dict] = []


# ------------------------------------------------------------ card ---------


def a4000() -> Dict[str, int]:
    out = subprocess.run(
        ["nvidia-smi", f"--id={CARD_UUID}",
         "--query-gpu=memory.used,memory.free,memory.total",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=20).stdout.strip()
    used, free, total = (int(x) for x in out.split(","))
    return {"used": used, "free": free, "total": total}


def image_server_on_card() -> bool:
    """AGENTS.md "one GPU consumer at a time": an image generator on the A4000
    (llama-swap's or another agent's own sd-server) means we pause, so
    neither side's numbers are the other's contention."""
    out = subprocess.run(
        ["nvidia-smi", f"--id={CARD_UUID}", "--query-compute-apps=process_name",
         "--format=csv,noheader"], capture_output=True, text=True, timeout=20).stdout
    return "sd-server" in out or "sd-cli" in out


def wait_for_room(max_wait_s: float = 3600) -> None:
    t0 = time.time()
    while True:
        m = a4000()
        if m["free"] >= HEADROOM_MIB and not image_server_on_card():
            if time.time() - t0 > 1:
                WAITS.append({"waited_s": round(time.time() - t0, 1), "free": m["free"]})
            return
        if time.time() - t0 > max_wait_s:
            raise SystemExit(f"A4000 free {m['free']} MiB < {HEADROOM_MIB} for "
                             f"{max_wait_s} s: not run (never evict)")
        time.sleep(10)


class PeakSampler:
    """Polls the A4000's used memory while a block runs (min free, max used)."""

    def __init__(self, every: float = 0.25):
        self.every, self.samples, self._stop = every, [], threading.Event()

    def __enter__(self):
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def _run(self):
        while not self._stop.is_set():
            try:
                self.samples.append(a4000())
            except Exception:                                    # noqa: BLE001
                pass
            time.sleep(self.every)

    def __exit__(self, *a):
        self._stop.set()
        self._t.join()

    def summary(self) -> Dict:
        if not self.samples:
            return {}
        return {"n": len(self.samples),
                "max_used": max(s["used"] for s in self.samples),
                "min_free": min(s["free"] for s in self.samples)}


# ------------------------------------------------------------ client -------


class Tev1:
    def __init__(self, url: str):
        self.url = url.rstrip("/")

    def _post(self, path: str, body: Dict, timeout: int = 120) -> Dict:
        req = urllib.request.Request(self.url + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)

    def props(self) -> Dict:
        with urllib.request.urlopen(self.url + "/props", timeout=30) as r:
            p = json.load(r)
        return {"model_path": p.get("model_path"),
                "n_ctx": (p.get("default_generation_settings") or {}).get("n_ctx"),
                "build": p.get("build_info")}

    def ntokens(self, text: str) -> int:
        return len(self._post("/tokenize", {"content": text})["tokens"])

    def decide(self, dec: Dict) -> Dict:
        body = dict(A.REQUEST_PARAMS, messages=A.messages(dec),
                    logprobs=True, top_logprobs=20)
        t0 = time.perf_counter()
        r = self._post("/v1/chat/completions", body)
        ms = (time.perf_counter() - t0) * 1000
        c = r["choices"][0]
        content = c["message"].get("content")
        lp = (((c.get("logprobs") or {}).get("content") or [{}])[0]).get("top_logprobs") or []
        return {"content": content, "finish": c.get("finish_reason"),
                "reasoning": c["message"].get("reasoning_content"),
                "key": A.letter_to_key(dec, content),
                "constrained": A.constrained_key(dec, lp),
                "top": [(t["token"], round(t["logprob"], 3)) for t in lp[:5]],
                "prompt_tokens": r["usage"]["prompt_tokens"],
                "completion_tokens": r["usage"]["completion_tokens"],
                "cached_tokens": (r["usage"].get("prompt_tokens_details") or {}).get("cached_tokens"),
                "ms": round(ms, 1),
                "server_ms": {k: round(v, 1) for k, v in (r.get("timings") or {}).items()
                              if k in ("prompt_ms", "predicted_ms")}}


def run_block(client: Tev1, decs: List[Dict], every: int = 25) -> List[Dict]:
    out = []
    for i, d in enumerate(decs):
        if i % every == 0:
            wait_for_room()
        out.append(client.decide(d))
    return out


def save(name: str, quant: str, blob: Dict) -> str:
    d = os.path.join(RESULTS, quant)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, f"{name}.json")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(blob, fh, indent=1)
    return p


def load(name: str, quant: str) -> Optional[Dict]:
    p = os.path.join(RESULTS, quant, f"{name}.json")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def pct(xs: List[float], p: float) -> float:
    s = sorted(xs)
    return round(s[min(len(s) - 1, max(0, int(round(p * (len(s) - 1)))))], 1)


# ------------------------------------------------------------ sanity -------


def cmd_sanity(args, client: Tev1) -> None:
    ex_dir = args.examples
    rows = []
    for f in sorted(os.listdir(ex_dir)):
        if not f.endswith(".json"):
            continue
        with open(os.path.join(ex_dir, f), encoding="utf-8") as fh:
            ex = json.load(fh)
        dec = {k: ex[k] for k in ("state", "question", "options")}
        r = client.decide(dec)
        rows.append({"example": f, **r})
    # the recommended request WITHOUT enable_thinking false, once
    dec = {k: ex[k] for k in ("state", "question", "options")}
    body = {"messages": A.messages(dec), "temperature": 0, "max_tokens": 8}
    raw = client._post("/v1/chat/completions", body)["choices"][0]
    blob = {"props": client.props(), "examples": rows,
            "thinking_left_on": {"content": raw["message"].get("content"),
                                 "reasoning": raw["message"].get("reasoning_content"),
                                 "finish": raw.get("finish_reason")}}
    print(json.dumps(blob, indent=1)[:3000])
    print("->", save("sanity", args.quant, blob))


# ------------------------------------------------------------ latency ------


def cmd_latency(args, client: Tev1) -> None:
    """n distinct states per size, each state new (no prefix reuse beyond the
    system prompt and the JSON opening); wall-clock round trip per decision."""
    # HARNESS FIX (PROTOCOL rule 3): the first version took windows of
    # docs/PROTOCOL.md at offsets (j*97) % (len - 3000); the file has fewer
    # than 3,001 words, so every offset was 0, all 20 states were identical,
    # and the server's prompt cache answered 2,000-token states in ~75 ms.
    # Now: a seeded random sample of words per row, and distinctness asserted.
    import random
    words = []
    for f in ("PROTOCOL.md", "CLM-EVAL.md", "SELF-IMPROVEMENT-PLAN.md"):
        with open(os.path.join(ROOT, "docs", f), encoding="utf-8") as fh:
            words += fh.read().split()
    q, opts = A.ROUTE_IN_VARIANTS["laya"]
    blob = {"props": client.props(), "sizes": {}}
    idle = a4000()
    blob["card_before"] = idle
    for target in (200, 600, 2000):
        decs, ntoks = [], []
        for j in range(args.n + 1):              # the extra one warms, not counted
            sample = random.Random(target * 1000 + j).sample(words, 2400)
            lo, hi = 10, 2400
            while lo < hi:                              # bisection on word count
                mid = (lo + hi) // 2
                st = A.render_state([("question", "Does TSL's Loop() accept an object form in r185?"),
                                     ("context", " ".join(sample[:mid]))])
                if client.ntokens(st) < target:
                    lo = mid + 1
                else:
                    hi = mid
            st = A.render_state([("question", "Does TSL's Loop() accept an object form in r185?"),
                                 ("context", " ".join(sample[:lo]))])
            ntoks.append(client.ntokens(st))
            decs.append(A.decision(st, q, opts))
        assert len({d["state"] for d in decs}) == len(decs), "latency states are not distinct"
        client.decide(decs.pop())                       # warm (not counted)
        ntoks.pop()
        wait_for_room()
        with PeakSampler() as ps:
            res = [client.decide(d) for d in decs]
        ms = [r["ms"] for r in res]
        blob["sizes"][str(target)] = {
            "n": len(res), "state_tokens": [min(ntoks), max(ntoks)],
            "prompt_tokens": [min(r["prompt_tokens"] for r in res),
                              max(r["prompt_tokens"] for r in res)],
            "p50_ms": pct(ms, 0.5), "p95_ms": pct(ms, 0.95), "max_ms": max(ms),
            "server_prompt_ms_p50": pct([r["server_ms"].get("prompt_ms", 0) for r in res], 0.5),
            "valid": sum(1 for r in res if r["key"] is not None),
            "cached_tokens_max": max((r["cached_tokens"] or 0) for r in res),
            "card": ps.summary()}
        print(target, blob["sizes"][str(target)])
    blob["card_after"] = a4000()
    print("->", save("latency", args.quant, blob))


# ------------------------------------------------------------ held-out -----


def _rows(path: str) -> List[Dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(x) for x in fh if x.strip()]


def cmd_heldout(args, client: Tev1) -> None:
    rows = _rows(HELD_OUT)
    arms = {"laya": dict(variant="laya"), "held": dict(variant="held"),
            "laya_reversed": dict(variant="laya", reverse=True)}
    # the other four orders of the three options (with laya and laya_reversed:
    # all six), for both wordings' order spread
    for j, o in enumerate([(0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1)]):
        arms[f"laya_order{j + 1}"] = dict(variant="laya", order=o)
    for j, o in enumerate([(0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]):
        arms[f"held_order{j + 1}"] = dict(variant="held", order=o)
    prev = load("heldout", args.quant) or {}
    blob = {"props": client.props(), "n": len(rows), "arms": dict(prev.get("arms") or {})}
    for name, kw in arms.items():
        if name in blob["arms"]:
            print(f"{name:14s} (kept from the previous run)")
            continue
        res = run_block(client, [A.route_in(r, **kw) for r in rows])
        blob["arms"][name] = [{"i": i, "key": r["key"], "constrained": r["constrained"],
                               "content": r["content"], "top": r["top"], "ms": r["ms"]}
                              for i, r in enumerate(res)]
        ok = sum(1 for r, x in zip(res, rows)
                 if (r["key"] == "investigate") == (x["label"] == "investigate"))
        print(f"{name:14s} investigate-vs-not {ok}/{len(rows)}  invalid "
              f"{sum(1 for r in res if r['key'] is None)}")
    blob["waits"] = WAITS
    print("->", save("heldout", args.quant, blob))


# ------------------------------------------------------------ route --------


def route_rows():
    """The 1,024 labelled corpus turns, the router's own prediction, and the
    facts the adapter needs -- exactly what bench/route/eval_route.py replays."""
    import tempfile
    tmp = tempfile.mkdtemp(prefix="tev1_route_")
    os.environ.setdefault("YAMADORI_CORPUS_DB", os.path.join(tmp, "corpus.sqlite3"))
    os.environ.setdefault("LLAMA_STACK_URL", "http://127.0.0.1:1")
    sys.path.insert(0, os.path.join(ROOT, "bench", "route"))
    import eval_route as ER
    labels = [r for r in ER.load_labels() if r.get("label")]
    ours = ER._our_names()
    pays = ER.corpus_payloads([r["id"] for r in labels])
    out = []
    for r in labels:
        p = pays.get(r["id"])
        if p is None:
            continue
        client = [n for n in p.get("tools_offered") or [] if n not in ours]
        rt = ER.route_of(ER.messages_of(p, r["ends_on"]), client, p.get("_repo"))
        out.append({"id": r["id"], "label": r["label"], "producer": r["producer"],
                    "ends_on": r["ends_on"], "router": rt["class"],
                    "readable": bool((rt.get("signals") or {}).get("readable")),
                    "payload": p, "client": client})
    return out


def cmd_route(args, client: Tev1) -> None:
    rows = route_rows()
    print(f"route rows: {len(rows)}")
    blob = {"props": client.props(), "n": len(rows), "rows": [], "arms": {}}
    blob["rows"] = [{k: r[k] for k in ("id", "label", "producer", "ends_on", "router", "readable")}
                    for r in rows]
    for name in args.arms.split(","):
        decs = [A.route_class(r["payload"], r["ends_on"], r["client"],
                              readable=r["readable"] if name == "facts" else None)
                for r in rows]
        res = run_block(client, decs)
        blob["arms"][name] = [{"key": x["key"], "constrained": x["constrained"],
                               "prompt_tokens": x["prompt_tokens"], "ms": x["ms"]} for x in res]
        acc = sum(1 for x, r in zip(res, rows) if x["key"] == r["label"])
        print(f"{name}: {acc}/{len(rows)} invalid {sum(1 for x in res if x['key'] is None)}")
    blob["waits"] = WAITS
    print("->", save("route", args.quant, blob))


# ------------------------------------------------------------ Phase 0.6 ----


def v0_steps() -> List[Dict]:
    """Per tool call of the Octopus V0 pilot: the struggle signals so far."""
    with open(os.path.join(V0, "hermes.jsonl"), encoding="utf-8") as fh:
        ev = [json.loads(x) for x in fh if x.strip()]
    steps, calls, errors, writes, repairs, recent = [], 0, 0, Counter(), 0, []
    last_call = None
    for e in ev:
        if e.get("type") == "text" and e.get("text", "").lstrip().startswith(
                ("Today I was inspired", "Repaired")) and "Repaired" in e["text"][:200]:
            repairs += 1
        if e.get("type") == "tool_use":
            calls += 1
            inp = e.get("input") or {}
            last_call = f"{e['name']} {inp.get('path') or inp.get('command', '')[:80]}"
            if e["name"] in ("write_file",) and inp.get("path"):
                writes[inp["path"]] += 1
        if e.get("type") == "tool_result":
            err = bool(e.get("is_error"))
            errors += err
            recent = (recent + [err])[-3:]
            steps.append({"step": calls, "tool_calls": calls, "tool_errors": errors,
                          "recent_errors": sum(recent),
                          "rewrites": ", ".join(f"{os.path.basename(p)} x{n}"
                                                for p, n in writes.items() if n > 1),
                          "reruns": 0, "repairs": repairs, "still_broken": False,
                          "last_call": last_call,
                          "last_result": ("ERROR " if err else "ok ") + str(e.get("output"))[:300]})
    return steps


def cmd_phase06(args, client: Tev1) -> None:
    steps = v0_steps()
    blob = {"props": client.props(), "n": len(steps), "labels": None,
            "note": "NO LABELS EXIST: smoke run of the adapters on real harness "
                    "signals; outputs are not scored",
            "struggle": [], "model": []}
    for s in steps:
        r = client.decide(A.struggle(s))
        m = client.decide(A.model_trigger(s))
        blob["struggle"].append({"step": s["step"], "errors": s["tool_errors"],
                                 "rewrites": s["rewrites"], "key": r["key"], "top": r["top"]})
        blob["model"].append({"step": s["step"], "key": m["key"], "top": m["top"]})
    # outcome: the one non-escalation with a failure in V0 (the patch at event 56)
    fail = next((s for s in steps if s["recent_errors"]), None)
    if fail:
        after = [f"step {s['step']}: {s['last_call']} -> {s['last_result'][:80]}"
                 for s in steps if s["step"] > fail["step"]][:4]
        o = client.decide(A.outcome(fail, after, ran=False))
        blob["outcome_after_patch_failure"] = {"step": fail["step"], "key": o["key"], "top": o["top"]}
    for k in ("struggle", "model"):
        print(k, Counter(x["key"] for x in blob[k]))
    print("outcome", blob.get("outcome_after_patch_failure"))
    print("->", save("phase06", args.quant, blob))


# ------------------------------------------------------------ control ------


def cmd_control(args) -> None:
    """E1 (docs/CLM-EVAL.md s.5): a logistic head on the Qwen3-Embedding-0.6B
    vectors we already serve, fitted with scripts/train_laya.py's own
    _fit_linear on the same 289 training rows, weight decay chosen by 5-fold
    CV (seed 0) inside the training rows only, then scored on the 120."""
    import numpy as np
    import code_search
    import train_laya as TL
    from laya_head import render_state

    tr = TL.load_labels("route_in")
    te = _rows(HELD_OUT)
    labels = ["investigate", "answer_directly", "clarify"]
    cache = os.path.join(RESULTS, "control_embeddings.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        Xtr, Xte = z["tr"], z["te"]
    else:
        def emb(texts):
            out = []
            for i in range(0, len(texts), 16):
                out.append(code_search.embed(texts[i:i + 16], is_query=True))
            return np.concatenate(out)
        Xtr = emb([r["_state"] for r in tr])
        Xte = emb([render_state("route_in", r) for r in te])
        os.makedirs(RESULTS, exist_ok=True)
        np.savez(cache, tr=Xtr, te=Xte)
    for X, name in ((Xtr, "train"), (Xte, "heldout")):   # PROTOCOL rule 1
        norms = np.linalg.norm(X, axis=1)
        if (norms < 0.5).any():
            raise SystemExit(f"{name}: {(norms < 0.5).sum()} embeddings are not unit-norm: dead")
    y = [labels.index(r["label"]) for r in tr]
    idx = list(range(len(y)))
    cv = TL.kfold(y, idx, 5, 0, None)
    scores = {}
    for wd in (1e-3, 1e-2, 1e-1, 1.0):
        accs = []
        for a, b in cv:
            W, bb = TL._fit_linear(Xtr[a].tolist(), [y[i] for i in a], 3, wd, seed=0)
            p = TL.predict_probs(W, bb, Xtr[b].tolist())
            accs.append(sum(1 for pp, i in zip(p, b) if int(np.argmax(pp)) == y[i]) / len(b))
        scores[wd] = sum(accs) / len(accs)
    wd = max(scores, key=lambda k: (scores[k], -k))
    W, bb = TL._fit_linear(Xtr.tolist(), y, 3, wd, seed=0)
    p = TL.predict_probs(W, bb, Xte.tolist())
    pred = [labels[int(np.argmax(pp))] for pp in p]
    ok = sum(1 for pr, r in zip(pred, te) if (pr == "investigate") == (r["label"] == "investigate"))
    blob = {"n_train": len(tr), "n_heldout": len(te), "cv": {str(k): round(v, 4) for k, v in scores.items()},
            "wd": wd, "binary": ok, "three_way": sum(1 for pr, r in zip(pred, te) if pr == r["label"]),
            "pred": pred}
    print({k: v for k, v in blob.items() if k != "pred"})
    with open(os.path.join(RESULTS, "control_e1.json"), "w", encoding="utf-8") as fh:
        json.dump(blob, fh, indent=1)


# ------------------------------------------------------------ report -------


def laya_live_signals(rows: List[Dict]) -> List[Dict]:
    """The live Laya signal per held-out row, rebuilt from the cached frozen
    features and the served head (live_route_check.txt: live == offline
    argmax on 120/120). Carries the head's abstain gate, which the live
    combination uses."""
    from laya_head import TrainedHead, render_state
    with open(os.path.join(ROOT, "index", "laya_staging_20260922_191425",
                           "features_heldout.json"), encoding="utf-8") as fh:
        fe = json.load(fh)["by_state"]
    h = TrainedHead.load(os.path.join(ROOT, "index", "laya", "route_in.json"))
    out = []
    for r in rows:
        d = h.decide(fe[render_state("route_in", r)])
        out.append({k: d.get(k) for k in ("choice", "margin", "abstain")})
    return out


def selection_with(rows: List[Dict], second: List[Optional[Dict]]) -> List[bool]:
    """selection.decide with a second signal, exactly as live_route_check.py
    calls it (tier max, the real symbol lookup, disagreement escalates)."""
    import tempfile
    tmp = tempfile.mkdtemp(prefix="tev1_sel_")
    os.environ.setdefault("YAMADORI_CORPUS_DB", os.path.join(tmp, "corpus.sqlite3"))
    os.environ.setdefault("LLAMA_STACK_URL", "http://127.0.0.1:1")
    import domains
    import selection
    import tiers
    t = tiers.resolve({"reasoning_effort": "max"}, tiers.from_header(None))
    dbs = selection.symbol_dbs()
    out = []
    for r, s in zip(rows, second):
        msgs = ([{"role": "user", "content": r["context"]}] if r.get("context") else []) \
            + [{"role": "user", "content": r["question"]}]
        gate = domains.tool_admission(msgs, None)
        if s is None:
            d = selection.decide(msgs, t, gate, dbs=dbs)
        else:
            d = selection.decide(msgs, t, gate, laya=s, laya_status="answered", dbs=dbs)
        out.append(bool(d["investigate"]))
    return out


def fmt(k: int, n: int) -> str:
    lo, hi = clopper_pearson(k, n)
    return f"{k}/{n} ({k / n:.3f}, 95% CI {lo:.3f}-{hi:.3f})"


def cmd_report(args) -> None:
    rows = _rows(HELD_OUT)
    y = [r["label"] for r in rows]
    yb = [lab == "investigate" for lab in y]
    hard = [i for i, r in enumerate(rows) if r.get("hard")]
    base = _rows(HELD_ROWS)
    assert [b["label"] for b in base] == y, "heldout_rows.jsonl is not aligned with the held-out file"
    live = laya_live_signals(rows)
    assert [d["choice"] for d in live] == [b["new"] for b in base], "Laya head rebuild differs"
    sel_alone = selection_with(rows, [None] * len(rows))
    sel_laya = selection_with(rows, live)

    def second(preds):
        return [{"choice": p, "abstain": p is None, "margin": None} for p in preds]

    corrb = {"laya_head": [(b["new"] == "investigate") == t for b, t in zip(base, yb)],
             "rule": [(b["rule"] == "investigate") == t for b, t in zip(base, yb)],
             "selection": [p == t for p, t in zip(sel_alone, yb)],
             "selection+laya": [p == t for p, t in zip(sel_laya, yb)]}
    # The recorded 89/120 (2026-09-22) is kept beside today's recomputation:
    # the package store and domains.derived_domains changed since, and every
    # combined arm below is computed with TODAY's selection, so it is paired.
    corrb["selection_recorded_0922"] = [b["selection_investigate"] == t
                                        for b, t in zip(base, yb)]
    esc = {"selection+laya": (sel_laya, sel_alone)}
    corr3 = {"laya_head": [b["new"] == t for b, t in zip(base, y)],
             "rule": [b["rule"] == t for b, t in zip(base, y)]}
    ctl = os.path.join(RESULTS, "control_e1.json")
    if os.path.exists(ctl):
        with open(ctl, encoding="utf-8") as fh:
            e1 = json.load(fh)
        corrb["E1_embed_head"] = [(p == "investigate") == t for p, t in zip(e1["pred"], yb)]
        corr3["E1_embed_head"] = [p == t for p, t in zip(e1["pred"], y)]
        s = selection_with(rows, second(e1["pred"]))
        corrb["selection+E1_embed_head"] = [p == t for p, t in zip(s, yb)]
        esc["selection+E1_embed_head"] = (s, sel_alone)
    tev_arms = []
    for quant in ("Q4_K_M", "Q8_0"):
        h = load("heldout", quant)
        if not h:
            continue
        for arm, res in h["arms"].items():
            if "_order" in arm:
                continue                        # reported in the order section
            for mode in ("key", "constrained"):
                name = f"tev1_{quant}_{arm}" + ("" if mode == "key" else "_constrained")
                preds = [r[mode] for r in res]
                corrb[name] = [(p == "investigate") == t for p, t in zip(preds, yb)]
                corr3[name] = [p == t for p, t in zip(preds, y)]
                if mode == "key":
                    tev_arms.append((name, preds, res))
                    s = selection_with(rows, second(preds))
                    corrb[f"selection+{name}"] = [p == t for p, t in zip(s, yb)]
                    esc[f"selection+{name}"] = (s, sel_alone)
    n = len(rows)
    print(f"\n## held-out route_in, n={n} ({sum(yb)} investigate), hard n={len(hard)}")
    print("\ninvestigate-vs-not:")
    for k, c in corrb.items():
        print(f"  {k:48s} {fmt(sum(c), n)}   hard {sum(c[i] for i in hard)}/{len(hard)}")
    print("\n3-way:")
    for k, c in corr3.items():
        print(f"  {k:48s} {fmt(sum(c), n)}")
    for name, preds, res in tev_arms:
        inv = sum(1 for r in res if r["key"] is None)
        dist = Counter(preds)
        missed = sum(1 for p, t in zip(preds, yb) if t and p != "investigate")
        unneeded = sum(1 for p, t in zip(preds, yb) if not t and p == "investigate")
        print(f"  {name}: invalid {inv}, dist {dict(dist)}, missed {missed}, unneeded {unneeded}")
    print("\nselection + second signal (disagreement escalates):")
    for k, (s, a) in esc.items():
        missed = sum(1 for p, t in zip(s, yb) if t and not p)
        unneeded = sum(1 for p, t in zip(s, yb) if p and not t)
        flips = sum(1 for p, q in zip(s, a) if p and not q)
        print(f"  {k:40s} {sum(corrb[k])}/{n}  missed {missed}  unneeded {unneeded}  flipped-to-investigate {flips}")
    print("\npaired exact McNemar, investigate-vs-not (a_only = first right, second wrong):")
    for name, _, _ in tev_arms:
        for ref in ("laya_head", "rule", "selection", "E1_embed_head"):
            if ref in corrb:
                m = mcnemar_exact(corrb[ref], corrb[name])
                print(f"  {ref:14s} vs {name:36s} {ref}-only {m['a_only']:3d}  "
                      f"tev1-only {m['b_only']:3d}  n_disc {m['a_only'] + m['b_only']:3d}  p = {m['p']:.4g}")
        s = f"selection+{name}"
        m = mcnemar_exact(corrb["selection+laya"], corrb[s])
        print(f"  {'selection+laya':14s} vs {s:36s} a-only {m['a_only']:3d}  b-only {m['b_only']:3d}  p = {m['p']:.4g}")
    if "E1_embed_head" in corrb:
        m = mcnemar_exact(corrb["laya_head"], corrb["E1_embed_head"])
        print(f"  laya_head vs E1_embed_head: laya-only {m['a_only']} e1-only {m['b_only']} p = {m['p']:.4g}")
    # order check: all six orders of the three options, per wording
    print("\noption-order spread (investigate-vs-not per order; six orders):")
    for quant in ("Q4_K_M", "Q8_0"):
        h = load("heldout", quant)
        if not h:
            continue
        for w in ("laya", "held"):
            names = [w] + [f"{w}_order{j}" for j in range(1, 6)]
            if w == "laya":
                names = ["laya", "laya_order1", "laya_order2", "laya_order3", "laya_order4", "laya_reversed"]
            names = [x for x in names if x in h["arms"]]
            if len(names) < 2:
                continue
            per = [sum((r["key"] == "investigate") == t for r, t in zip(h["arms"][x], yb)) for x in names]
            per3 = [sum(r["key"] == t for r, t in zip(h["arms"][x], y)) for x in names]
            allsame = sum(1 for i in range(n) if len({h["arms"][x][i]["key"] for x in names}) == 1)
            vote = []
            for i in range(n):
                c = Counter(h["arms"][x][i]["key"] for x in names)
                vote.append(c.most_common(1)[0][0])
            vb = [(p == "investigate") == t for p, t in zip(vote, yb)]
            m = mcnemar_exact(corrb["laya_head"], vb)
            me = mcnemar_exact(corrb["E1_embed_head"], vb) if "E1_embed_head" in corrb else None
            print(f"  {quant} {w}: orders {len(names)}  binary per order {per}  "
                  f"(min {min(per)}, mean {sum(per) / len(per):.1f}, max {max(per)})  3-way {per3}  "
                  f"all orders agree on {allsame}/{n}  majority vote {sum(vb)}/{n} "
                  f"[vs laya p={m['p']:.3g} ({m['a_only']}/{m['b_only']})"
                  + (f", vs E1 p={me['p']:.3g} ({me['a_only']}/{me['b_only']})" if me else "") + "]")
            worst = names[per.index(min(per))]
            wb = [(r["key"] == "investigate") == t for r, t in zip(h["arms"][worst], yb)]
            mw = mcnemar_exact(corrb["laya_head"], wb)
            print(f"      worst order ({worst}) vs laya_head: p={mw['p']:.3g} ({mw['a_only']}/{mw['b_only']})")
    q4, q8 = load("heldout", "Q4_K_M"), load("heldout", "Q8_0")
    if q4 and q8:
        for arm in q4["arms"]:
            same = sum(1 for a, b in zip(q4["arms"][arm], q8["arms"][arm]) if a["key"] == b["key"])
            print(f"  quant agreement {arm}: Q4_K_M vs Q8_0 same key {same}/{n}")

    # route classes
    for quant in ("Q4_K_M", "Q8_0"):
        r = load("route", quant)
        if not r:
            continue
        rows_r = r["rows"]
        nr = len(rows_r)
        router_ok = [x["router"] == x["label"] for x in rows_r]
        print(f"\n## route classes {quant}, n={nr} (in-sample for the router's rules, not for Tev1)")
        print(f"  router (mcp/route.py)            {fmt(sum(router_ok), nr)}")
        for arm, res in r["arms"].items():
            ok = [x["key"] == row["label"] for x, row in zip(res, rows_r)]
            inv = sum(1 for x in res if x["key"] is None)
            m = mcnemar_exact(router_ok, ok)
            code = {"code_edit", "code_generation"}
            mis = sum(1 for x, row in zip(res, rows_r)
                      if row["label"] in ("utility", "agent_step") and x["key"] in code)
            codefn = sum(1 for x, row in zip(res, rows_r) if row["label"] in code and x["key"] not in code)
            print(f"  tev1 {arm:6s} {fmt(sum(ok), nr)}  invalid {inv}  misroutes->code {mis}  "
                  f"code missed {codefn}  McNemar router-only {m['a_only']} tev1-only {m['b_only']} p={m['p']:.3g}")
            per = {}
            for c in ("utility", "agent_step", "code_edit", "code_generation", "library_question", "prose"):
                tp = sum(1 for x, row in zip(res, rows_r) if row["label"] == c and x["key"] == c)
                nn = sum(1 for row in rows_r if row["label"] == c)
                pp = sum(1 for x in res if x["key"] == c)
                per[c] = f"R {tp}/{nn} P {tp}/{pp}"
            print("   ", per)
            byp = {}
            for prod in sorted({row["producer"] for row in rows_r}):
                idx = [i for i, row in enumerate(rows_r) if row["producer"] == prod]
                byp[prod] = f"{sum(ok[i] for i in idx)}/{len(idx)} (router {sum(router_ok[i] for i in idx)})"
            print("    by producer:", byp)
            us = [i for i, row in enumerate(rows_r) if row["ends_on"] == "user"]
            print(f"    user-speaking only: tev1 {sum(ok[i] for i in us)}/{len(us)}, router {sum(router_ok[i] for i in us)}/{len(us)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["sanity", "latency", "heldout", "route", "phase06",
                                    "control", "report"])
    ap.add_argument("--quant", default="Q4_K_M")
    ap.add_argument("--url", default="http://127.0.0.1:10091")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--arms", default="text,facts")
    ap.add_argument("--examples", default=r"C:\Users\jwals\textgen\user_data\models\tev1-4b\tev1-repo\examples")
    args = ap.parse_args()
    if args.cmd == "control":
        return cmd_control(args)
    if args.cmd == "report":
        return cmd_report(args)
    client = Tev1(args.url)
    p = client.props()
    if args.quant not in (p.get("model_path") or ""):
        raise SystemExit(f"server at {args.url} serves {p.get('model_path')}, not {args.quant}")
    {"sanity": cmd_sanity, "latency": cmd_latency, "heldout": cmd_heldout,
     "route": cmd_route, "phase06": cmd_phase06}[args.cmd](args, client)


if __name__ == "__main__":
    main()
