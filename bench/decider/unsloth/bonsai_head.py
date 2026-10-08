#!/usr/bin/env python
"""PHASE 4: A CLEF-STYLE DECISION HEAD ON A FROZEN BONSAI (small-scale feasibility).

The operator's question (2026-10-07): "can we do this with one of our models" -- fit Bonsai and a smaller decider on one
card, or better, use the model already in the stack. Bonsai is a ternary PTQ GGUF: no LoRA is possible, so the only
trainable thing is a HEAD on its frozen hidden states. Unsloth's recipe has that stage (`freeze_backbone`: "Stage A of
the from-LM recipe: train only the decision head on frozen backbone features") and Cloudflare's head reads exactly the
final hidden states, so the test is:

  1. records     encode decisions as Clef records (the package's own `encode_record`: state + typed questions, spans of every
                 question and option) with the Qwen3.5 tokenizer, which IS Bonsai's (all 248,077 real tokens and all 247,587
                 merges equal the GGUF's; only [PAD...] filler ids differ) -> records_<set>.pkl
  2. extract     llama-server (our engine, the Bonsai GGUF, --embeddings --pooling none) returns the final-norm hidden state
                 of EVERY token of a record (/embedding); stored fp16, one .npy per record. The server is a process THIS
                 script's caller starts on a free card (never llama-swap's); see start_server().
  3. train       Cloudflare-style JointSchemaHead (width 1024, 2 routing + 4 decoder layers, 16 heads, ff 4096: 128.1M parameters
                 at hidden 5120), soft cross-entropy, head lr 3e-4, on the cached states. `lexical` features read the model's
                 `output.weight` rows (PTQ1_0, decoded with scripts/extract_token_embd.py's decoder, Hadamard-rotated as stored).
  4. eval        the same held-out decisions as the 2B run (typed-decisions test, BANKING77, CLINC150), JevBench's public items,
                 and our skill cases, against jjava's letter read on the same items.

Usage (each step is separate so a crash costs one step):
    python bonsai_head.py records --set guide_train --n 3000
    python bonsai_head.py records --set guide_test  --n-typed 400 --n-banking 150 --n-clinc 60
    python bonsai_head.py records --set jevbench
    python bonsai_head.py records --set skills
    python bonsai_head.py extract --set guide_train --server http://127.0.0.1:10031
    python bonsai_head.py train --sets guide_train --epochs 3 --out RUN
    python bonsai_head.py eval  --head RUN --sets guide_test jevbench skills
CPU-testable without Bonsai: `selftest` (random hidden states through the same code path).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import pickle
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DATA = Path(os.environ.get("YAMADORI_DECIDER_DATA", ROOT / "index" / "decider_train"))
STATES = DATA / "bonsai_states"
GGUF = "C:/Users/jwals/textgen/user_data/models/Ternary-Bonsai-2-27B-PTQ1_0.gguf"
TOKENIZER = "unsloth/Qwen3.5-2B"          # same vocabulary as Bonsai (checked against the GGUF's token table)
HIDDEN = 5120
MAX_LEN = 4096
os.environ.setdefault("HF_HOME", "C:/Users/jwals/textgen/user_data/models/hf-unsloth")
VENDOR = Path(sys.prefix) / "Lib" / "site-packages" / "unsloth" / "_vendor" / "clef" / "joint_schema_model.py"


def load_ref():
    spec = importlib.util.spec_from_file_location("clef_ref", str(VENDOR))
    m = importlib.util.module_from_spec(spec)
    sys.modules["clef_ref"] = m
    spec.loader.exec_module(m)
    return m


def clef_q(q: dict) -> dict:
    out = {"type": q["type"], "instructions": q.get("instructions")}
    if q.get("criteria") is not None:
        out["criteria"] = q["criteria"]
    return out


def target_of(ref, q: dict, gold) -> tuple[list, int]:
    """(soft target over the question's options in the encoder's option order, argmax index)."""
    keys = [k for k, _ in ref.question_options(q)]
    if not isinstance(gold, dict):
        gold = {"label": gold}
    label = gold.get("label")
    kind = q["type"]
    if kind == "noul":
        label = str(label).strip().lower() if label is not None else None
        p = gold.get("probabilities") or ({"true": gold["noul"]} if isinstance(gold.get("noul"), (int, float)) else None)
        if isinstance(p, dict) and ("true" in p or "false" in p):
            t = float(p["true"]) if "true" in p else 1.0 - float(p["false"])
            return [1.0 - t if k == "false" else t for k in keys], int(t >= 0.5 and keys.index("true") or keys.index("false"))
    elif kind == "score" and label is not None and not isinstance(label, bool):
        label = str(int(float(label)))
    elif label is not None:
        label = str(label)
    p = gold.get("probabilities")
    if isinstance(p, dict):
        v = [max(0.0, float(p.get(k, 0.0))) for k in keys]
        s = sum(v)
        if s > 0:
            v = [x / s for x in v]
            return v, v.index(max(v))
    if label in keys:
        return [1.0 if k == label else 0.0 for k in keys], keys.index(label)
    raise ValueError("no usable gold")


# --------------------------------------------------------------------------------------------- records
def read_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def build_records(rows: list[dict], tok, ref, name: str) -> list[dict]:
    out = []
    for i, r in enumerate(rows):
        try:
            qs, tg, lab = {}, [], []
            for qid, q in r["questions"].items():
                cq = clef_q(q)
                t, y = target_of(ref, cq, r["gold"][qid])
                qs[str(qid)] = cq
                tg.append(t)
                lab.append(y)
            rec = ref.encode_record(tok, {"state": r["state"], "questions": qs}, max_length=MAX_LEN)
        except Exception as e:  # skipped rows are counted, never hidden
            out.append({"id": f"{name}:{i}", "skipped": str(e)[:120], "source": r.get("source")})
            continue
        out.append({"id": f"{name}:{i}", "source": r.get("source"), "record": rec, "targets": tg, "labels": lab,
                    "qtypes": [q["type"] for q in qs.values()], "n_tokens": len(rec.input_ids), "meta": r.get("meta")})
    return out


def cmd_records(a) -> int:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    ref = load_ref()
    rows: list[dict] = []
    if a.set == "guide_train":
        rows = read_jsonl(DATA / "train.jsonl")[: a.n]
    elif a.set == "guide_test":
        allr = read_jsonl(DATA / "test.jsonl")
        for src, n in (("typed_decisions", a.n_typed), ("banking77", a.n_banking), ("clinc150", a.n_clinc)):
            rows += [r for r in allr if r["source"] == src][:n]
    elif a.set == "jevbench":
        sys.path.insert(0, str(HERE))
        import eval_jevbench as EJ
        sys.path.insert(0, EJ.CLONE)
        from jevbench import tasks
        for fn, tier in (("easy.jsonl", "easy"), ("original.jsonl", "standard"), ("hard.jsonl", "hard")):
            for t in tasks.load_jsonl(os.path.join(EJ.CLONE, "datasets", "public", fn)):
                q = EJ.clef_question(t)
                labels = [str(x) for x in t.labels]
                gold = ({"label": "true" if t.expected in (True, "yes", "true") else "false"} if q["type"] == "noul"
                        else {"label": str(t.expected)})
                rows.append({"source": "jevbench:" + tier, "state": t.state, "questions": {"decision": q},
                             "gold": {"decision": gold}, "meta": {"task_id": t.id, "tier": tier, "labels": labels}})
    elif a.set == "skills":
        # one record per case: every item is two noul questions (relevance, need), scored jointly as Clef does
        items = read_jsonl(DATA / "skills_items.jsonl")
        by: dict = {}
        for it in items:
            by.setdefault(it["case"], []).append(it)
        from eval_skills import REL, NEED
        cases = sorted(by)
        random.Random(11).shuffle(cases)
        if a.n_cases:
            cases = cases[: a.n_cases]
        for case in cases:
            its = by[case]
            qs, gold = {}, {}
            for it in its:
                for nm, text, truth in (("rel", REL, it["level"] >= 1), ("need", NEED, it["level"] == 3)):
                    k = f"{nm}:{it['key']}"
                    qs[k] = {"type": "noul", "instructions": f"{text}\n\nFact: {it['fact']}"}
                    gold[k] = {"label": "true" if truth else "false"}
            rows.append({"source": "skills", "state": its[0]["state"], "questions": qs, "gold": gold,
                         "meta": {"case": case, "kind": its[0]["kind"], "keys": [it["key"] for it in its]}})
    recs = build_records(rows, tok, ref, a.set)
    ok = [r for r in recs if "record" in r]
    pickle.dump(recs, open(DATA / f"bonsai_records_{a.set}.pkl", "wb"))
    toks = [r["n_tokens"] for r in ok]
    print(a.set, "records", len(ok), "skipped", len(recs) - len(ok),
          "tokens total", sum(toks), "max", max(toks) if toks else 0,
          "decisions", sum(len(r["targets"]) for r in ok),
          "fp16 states on disk ~%.1f GB" % (sum(toks) * HIDDEN * 2 / 1e9))
    return 0


# --------------------------------------------------------------------------------------------- extract
def start_server_cmd(port: int = 10031, ctx: int = 8192) -> str:
    """The command a caller runs on a FREE card (never through llama-swap). -b = -ub = ctx: a non-pooled embeddings
    request must fit one physical batch."""
    eng = "C:/Users/jwals/engines/llama-bonsai2-ada-cand0042/src/build/bin/llama-server.exe"
    return (f'{eng} -m {GGUF} -dev CUDA0 -ngl 999 -c {ctx} -b {ctx} -ub {ctx} -fa on --embeddings --pooling none '
            f'--port {port} --no-warmup --jinja -np 1')


def post_embed(server: str, ids: list[int], timeout=600):
    import urllib.request
    req = urllib.request.Request(server + "/embedding", data=json.dumps({"content": ids}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = json.loads(r.read())
    import numpy as np
    emb = body[0]["embedding"] if isinstance(body, list) else body["embedding"]
    return np.asarray(emb, dtype=np.float32)


def cmd_extract(a) -> int:
    """Two client threads against the single-slot server: while one parses the (huge, ~100 KB per token) JSON the other's request is
    already prefilling."""
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor
    load_ref()      # the pickled records are clef_ref dataclasses
    recs = pickle.load(open(DATA / f"bonsai_records_{a.set}.pkl", "rb"))
    out = STATES / a.set
    out.mkdir(parents=True, exist_ok=True)
    todo = [r for r in recs if "record" in r and not (out / (r["id"].replace(":", "_") + ".npy")).exists()]
    t0 = time.time()
    done = [0, 0]

    def one(r):
        ids = list(r["record"].input_ids)
        h = post_embed(a.server, ids)
        if h.shape != (len(ids), HIDDEN):
            raise SystemExit(f"{r['id']}: got {h.shape}, want {(len(ids), HIDDEN)} (is the server --embeddings --pooling none?)")
        np.save(out / (r["id"].replace(":", "_") + ".npy"), h.astype(np.float16))
        done[0] += 1
        done[1] += len(ids)
        if done[0] % 25 == 0:
            print(done[0], "of", len(todo), "records", done[1], "tokens", round(time.time() - t0), "s", round(done[1] / (time.time() - t0)), "tok/s", flush=True)

    with ThreadPoolExecutor(2) as ex:
        list(ex.map(one, todo))
    print("extracted", done[0], "records,", done[1], "tokens in", round(time.time() - t0), "s ->", out)
    return 0


# --------------------------------------------------------------------------------------------- the head
def lm_head_rows(token_ids: set[int]):
    """Rows of Bonsai's `output.weight` (PTQ1_0, Hadamard-rotated as stored) for the given token ids, fp16."""
    import numpy as np
    sys.path.insert(0, str(ROOT / "scripts"))
    import extract_token_embd as X
    meta, tensors, start = X.read_header(GGUF)
    (n_cols, n_rows), ttype, off = tensors["output.weight"]
    assert ttype == X.PTQ1_0 and n_cols == HIDDEN
    ids = sorted(token_ids)
    out = {}
    per_row = n_cols // X.BLOCK_WEIGHTS * X.BLOCK_BYTES
    mm = np.memmap(GGUF, dtype=np.uint8, mode="r", offset=start + off, shape=(n_rows * per_row,))
    for i in ids:
        raw = np.asarray(mm[i * per_row:(i + 1) * per_row]).reshape(-1, X.BLOCK_BYTES)
        out[i] = X.decode_blocks(raw).reshape(n_cols).astype(np.float16)
    return out


class Sparse:
    """A stand-in for the [vocab, hidden] output-embedding matrix that only holds the rows the batch indexes."""

    def __init__(self, rows: dict, dev, dtype):
        import torch
        self.map = {k: torch.tensor(v) for k, v in rows.items()}
        self.dev, self.dtype = dev, dtype

    def __getitem__(self, idx):
        import torch
        flat = idx.reshape(-1).tolist()
        w = torch.stack([self.map[i] for i in flat]).to(self.dev, self.dtype)
        return w.reshape(*idx.shape, -1)


def make_head(dev):
    import torch
    cfg = dict(hidden_size=HIDDEN, width=1024, routing_layers=2, layers=4, heads=16, feedforward=4096)
    try:
        from unsloth.models.clef import JointSchemaHead
    except Exception:
        JointSchemaHead = load_ref().JointSchemaHead
    return JointSchemaHead(**cfg).to(dev, torch.float32), cfg


def batches(recs, rng, bs, shuffle=True):
    idx = list(range(len(recs)))
    if shuffle:
        rng.shuffle(idx)
    idx.sort(key=lambda i: recs[i]["n_tokens"] // 64)       # length-grouped, then shuffled in blocks
    blocks = [idx[i:i + bs] for i in range(0, len(idx), bs)]
    if shuffle:
        rng.shuffle(blocks)
    return blocks


def forward(head, ref, batch, sets_dir, lex, dev, synthetic=False):
    import numpy as np
    import torch
    L = max(r["n_tokens"] for r in batch)
    hid = torch.zeros(len(batch), L, HIDDEN, dtype=torch.float16)
    ids = torch.zeros(len(batch), L, dtype=torch.long)
    mask = torch.zeros(len(batch), L, dtype=torch.long)
    for i, r in enumerate(batch):
        n = r["n_tokens"]
        if synthetic:
            hid[i, :n] = torch.randn(n, HIDDEN).half()
        else:
            hid[i, :n] = torch.from_numpy(np.load(sets_dir[r["set"]] / (r["id"].replace(":", "_") + ".npy")))
        ids[i, :n] = torch.tensor(list(r["record"].input_ids))
        mask[i, :n] = 1
    hid = hid.to(dev)
    if not hasattr(head, "forward_per_record"):      # Cloudflare's reference head wants float32
        hid = hid.float()
    out = head(hid, ids.to(dev), mask.to(dev), [r["record"] for r in batch], lex)
    flat = [q for rec in out for q in rec]
    return flat


def load_sets(names):
    load_ref()
    recs = []
    dirs = {}
    for s in names:
        for r in pickle.load(open(DATA / f"bonsai_records_{s}.pkl", "rb")):
            if "record" in r:
                r["set"] = s
                recs.append(r)
        dirs[s] = STATES / s
    return recs, dirs


def train(a, synthetic=False) -> dict:
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ref = load_ref()
    recs, dirs = load_sets(a.sets)
    if synthetic:
        recs = recs[: a.limit or 8]
    rng = random.Random(3407)
    toks = {t for r in recs for q in r["record"].questions for s, e in q.option_spans for t in r["record"].input_ids[s:e]}
    lex = Sparse(lm_head_rows(toks) if not synthetic else {t: torch.randn(HIDDEN).half().numpy() for t in toks}, dev, torch.float32)
    head, cfg = make_head(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=0.01)
    steps_total = max(1, a.epochs * math.ceil(len(recs) / a.batch))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 5) * max(0.0, 1 - s / steps_total))
    log, step, t0 = [], 0, time.time()
    head.train()
    for ep in range(a.epochs):
        for b in batches(recs, rng, a.batch):
            batch = [recs[i] for i in b]
            flat = forward(head, ref, batch, dirs, lex, dev, synthetic)
            loss = 0.0
            nq = 0
            for z, t in zip(flat, [t for r in batch for t in r["targets"]]):
                tt = torch.tensor(t, device=dev)
                loss = loss - (tt * torch.log_softmax(z.float(), -1)).sum()
                nq += 1
            loss = loss / nq
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 10 == 0:
                log.append({"step": step, "loss": round(float(loss), 4)})
                print(step, steps_total, round(float(loss), 4), flush=True)
    res = {"steps": step, "epochs": a.epochs, "seconds": round(time.time() - t0, 1), "log": log[::max(1, len(log) // 40)],
           "params": sum(p.numel() for p in head.parameters()), "head_config": cfg, "records": len(recs)}
    if a.out and not synthetic:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        torch.save(head.state_dict(), Path(a.out) / "head.pt")
        (Path(a.out) / "train.json").write_text(json.dumps(res, indent=1))
    return res


def cmd_eval(a) -> int:
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ref = load_ref()
    recs, dirs = load_sets(a.sets)
    toks = {t for r in recs for q in r["record"].questions for s, e in q.option_spans for t in r["record"].input_ids[s:e]}
    lex = Sparse(lm_head_rows(toks), dev, torch.float32)
    head, _ = make_head(dev)
    head.load_state_dict(torch.load(Path(a.head) / "head.pt", map_location=dev))
    head.eval()
    out = []
    with torch.no_grad():
        for b in batches(recs, random.Random(0), a.batch, shuffle=False):
            batch = [recs[i] for i in b]
            flat = forward(head, ref, batch, dirs, lex, dev)
            k = 0
            for r in batch:
                for qi, qid in enumerate(r["record"].questions):
                    p = torch.softmax(flat[k].float(), -1).cpu().tolist()
                    out.append({"id": r["id"], "set": r["set"], "source": r["source"], "q": qid.question_id,
                                "options": list(qid.option_ids), "p": p, "label": r["labels"][qi],
                                "target": r["targets"][qi], "meta": r.get("meta")})
                    k += 1
    p = Path(a.head) / ("eval_" + "+".join(a.sets) + ".jsonl")
    p.write_text("\n".join(json.dumps(x) for x in out), encoding="utf-8")
    by = {}
    for x in out:
        by.setdefault(x["source"], []).append(int(max(range(len(x["p"])), key=x["p"].__getitem__) == x["label"]))
    print({s: (round(sum(v) / len(v), 4), len(v)) for s, v in by.items()})
    post_eval(a, out)
    return 0


def post_eval(a, out) -> None:
    """JevBench items scored by JevBench's own code; skills as pred rows for skills_score.py."""
    if "jevbench" in a.sets:
        sys.path.insert(0, str(HERE))
        import eval_jevbench as EJ
        sys.path.insert(0, EJ.CLONE)
        from jevbench import scoring, tasks
        by_id = {}
        for fn in ("easy.jsonl", "original.jsonl", "hard.jsonl"):
            for t in tasks.load_jsonl(os.path.join(EJ.CLONE, "datasets", "public", fn)):
                by_id[t.id] = t
        recs = []
        for x in out:
            if x["set"] != "jevbench":
                continue
            t = by_id[x["meta"]["task_id"]]
            if t.question["type"] == "noul":
                pt = dict(zip(x["options"], x["p"]))["true"]
                probs = {"yes": pt, "no": 1 - pt}
            else:
                probs = dict(zip(x["options"], x["p"]))
            sc = scoring.score_task(probs, t)
            recs.append({"task_id": t.id, "tier": x["meta"]["tier"], "family": t.family, "type": t.question["type"], "group": t.group,
                         "n_labels": len(t.labels), "ok": True, "latency_s": None, "valid": sc["valid"], "strict_valid": sc.get("strict_valid"),
                         "correct": sc["correct"], "predicted": sc.get("predicted"), "ordinal_ev": sc.get("ordinal_ev"), "probs": sc.get("probs")})
        d = Path(a.head) / "jevbench"
        d.mkdir(exist_ok=True)
        (d / "items.jsonl").write_text(chr(10).join(json.dumps(r) for r in recs), encoding="utf-8")
        print("jevbench items", len(recs), "-> summarize with eval_jevbench.py --summarize", d)
    if "skills" in a.sets:
        pred = {}
        for x in out:
            if x["set"] != "skills":
                continue
            nm, key = x["q"].split(":", 1)
            pred.setdefault(key, {"case": x["meta"]["case"], "key": key})[{"rel": "relevance", "need": "need"}[nm]] = dict(zip(x["options"], x["p"]))["true"]
        (Path(a.head) / "skills_pred.jsonl").write_text(chr(10).join(json.dumps(v) for v in pred.values()), encoding="utf-8")
        print("skills pred rows", len(pred))


def cmd_selftest(a) -> int:
    """CPU: synthetic hidden states through the same records -> head -> loss path (no Bonsai, no GPU)."""
    a.sets, a.limit = ["guide_test"], 6
    a.epochs, a.batch, a.lr, a.out = 1, 2, 1e-4, None
    r = train(a, synthetic=True)
    print("selftest ok", {k: r[k] for k in ("steps", "params", "records")})
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("records"); p.add_argument("--set", required=True)
    p.add_argument("--n", type=int, default=3000); p.add_argument("--n-typed", type=int, default=400)
    p.add_argument("--n-cases", type=int, default=0, help="skills: only this many cases (seeded shuffle)")
    p.add_argument("--n-banking", type=int, default=150); p.add_argument("--n-clinc", type=int, default=60)
    p = sub.add_parser("extract"); p.add_argument("--set", required=True); p.add_argument("--server", default="http://127.0.0.1:10031")
    p = sub.add_parser("train"); p.add_argument("--sets", nargs="+", required=True); p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch", type=int, default=4); p.add_argument("--lr", type=float, default=3e-4); p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=0)
    p = sub.add_parser("eval"); p.add_argument("--head", required=True); p.add_argument("--sets", nargs="+", required=True); p.add_argument("--batch", type=int, default=4)
    sub.add_parser("selftest")
    sub.add_parser("server-cmd")
    a = ap.parse_args()
    if a.cmd == "records":
        return cmd_records(a)
    if a.cmd == "extract":
        return cmd_extract(a)
    if a.cmd == "train":
        r = train(a)
        print(json.dumps({k: r[k] for k in ("steps", "seconds", "params")}))
        return 0
    if a.cmd == "eval":
        return cmd_eval(a)
    if a.cmd == "server-cmd":
        print(start_server_cmd())
        return 0
    return cmd_selftest(a)


if __name__ == "__main__":
    sys.exit(main())
