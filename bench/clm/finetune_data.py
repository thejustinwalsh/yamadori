#!/usr/bin/env python
"""The CLM skill-selection fine-tune: data, cached encoder embeddings, and the
exact training command. Nothing here trains. (docs/CLM.md "Fine-tune plan".)

    python bench/clm/finetune_data.py build          # parquet + manifest
    python bench/clm/finetune_data.py embed --url http://127.0.0.1:PORT
    python bench/clm/finetune_data.py command        # prints the finetune.py line

THE RECIPE is the official one, pinned: Contrastive-LM/CLM @ bb42c6c5,
train/finetune.py --task choice (typed System One questions; each question's
state text and candidate texts from clm.schema.build_pairs; the heads warm-
started from CLM_v0.1-8B.pt; bidirectional in-batch InfoNCE over the batch's
distinct option texts; the best epoch by validation accuracy). Only the heads
train; the encoder is frozen, so its embeddings are computed ONCE -- by OUR
served quantized encoder (the one decide() uses at request time, not vLLM
bf16) -- and handed to finetune.py through its own embedding cache
(`TextCache`: sha1(text) -> float16 vector, file
choice_<slug(embed_model)>_<max_len>.npz). With every text cached,
finetune.py never builds an encoder.

THE DATA (--workflow skills; rows in typed-decisions form: id, state,
questions, gold):

  TRAIN (split "train"; finetune.py holds out --val-frac of it by row id)
    * activation tests, index/skills/library/*/tests.json, every armed skill:
      each `should` case -> the skill's trigger text (its SKILL.md
      description) is the gold option; each `should_not` case (the near
      misses first) -> "none" is gold. Options: the skill, up to
      HARD_NEGATIVES skills of the SAME AREA (a shared framework, else a
      shared language, seeded), and the "none" option (clm.NONE_OPTION).
    * client-traffic selection logs: skill_learn's labels (router_labels.
      jsonl) with traffic == "client" ONLY (test and unknown rows are never
      learned from -- skill_learn's own rule), grouped per fallback: the
      request excerpt is the state, its candidates (by id, at the skill's
      CURRENT trigger text) plus "none" the options, the used skills the
      gold (split evenly), "none" when the fallback used nothing.
  TEST (split "test") = THE DAILY EVAL, HELD OUT
    * bench/skills/daily_eval.jsonl rows with a `user` text whose `must`
      names a skill that exists (gold: that skill) or that expect nothing
      (gold: none), against the same option construction as the fidelity
      set. Area-only rows have no single gold option and are left out
      (counted). Test material (`source` rows: the Octopus spec, the pagoda
      prompt) is read only to screen train rows and never written.
  HELD OUT MEANS HELD OUT: a train state that shares any 8-word shingle with
  any daily-eval text (user, turns, append, and the `source` material) is
  dropped and counted.

The instruction is clm.INSTRUCTIONS for every question.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "mcp"))
sys.path.insert(0, HERE)

import clm  # noqa: E402
import skill_learn  # noqa: E402  (before replay_selection: it points the stores elsewhere)

LABELS = skill_learn.LABELS

OUT = os.environ.get("YAMADORI_CLM_FT_DIR",
                     os.path.join(ROOT, "index", "clm", "finetune"))
DATA = os.path.join(OUT, "data")
WORKFLOW = "skills"
EMBED_MODEL = "Qwen/Qwen3-8B"          # finetune.py --embed-model (a cache key)
MAX_LEN = 2048                          # finetune.py --max-len (choice default)
HARD_NEGATIVES = 5
SEED = 20260927
SHINGLE = 8
CLM_REPO = "https://github.com/Contrastive-LM/CLM"
CLM_COMMIT = "bb42c6c5bf914fd449bed2f6ca65be80602cb1f7"
MODELS_DIR = clm.MODELS_DIR


def _slug(s: str) -> str:            # finetune.py _slug, verbatim
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")


def cache_path() -> str:
    return os.path.join(OUT, "embeddings",
                        f"choice_{_slug(EMBED_MODEL)}_{MAX_LEN}.npz")


def _shingles(text: str) -> set[str]:
    w = re.findall(r"[a-z0-9]+", (text or "").lower())
    return {" ".join(w[i:i + SHINGLE]) for i in range(max(0, len(w) - SHINGLE + 1))}


def _held_out_texts() -> list[str]:
    import replay_selection as rs  # bench/skills
    out = []
    for r in rs._daily_rows():
        try:
            out.append(rs._user_text(r))
        except Exception:                                        # noqa: BLE001
            pass
        for t in r.get("turns") or []:
            out.append(t.get("user") or "")
            st = t.get("step") or {}
            out.append(str(st.get("result") or ""))
    return [t for t in out if t]


def _row(rid: str, state: str, keys: list[str], texts: list[str],
         gold: dict[str, float], source: str) -> dict:
    crit = dict(zip(keys, texts))
    label = max(gold, key=gold.get)
    return {"id": rid, "workflow": WORKFLOW, "source": source,
            "state": state,
            "questions": json.dumps({"skill": {
                "type": "choice", "instructions": clm.INSTRUCTIONS,
                "criteria": crit}}, ensure_ascii=False),
            "gold": json.dumps({"skill": {"label": label,
                                          "probabilities": gold}})}


def build() -> dict:
    sys.path.insert(0, os.path.join(ROOT, "bench", "skills"))
    import fidelity
    rng = random.Random(SEED)
    skills = fidelity._skills()
    by_name = {s["name"]: s for s in skills}
    ids = {}
    import skill_md
    for folder in skill_md.folders(fidelity.LIBRARY):
        sk, _t = skill_md.read_folder(folder)
        y = sk.get("yamadori") or {}
        if y.get("id") and sk.get("name") in by_name:
            ids[y["id"]] = sk["name"]
    held = set()
    for t in _held_out_texts():
        held |= _shingles(t)
    stats = Counter()

    def leaks(text: str) -> bool:
        return bool(_shingles(text) & held)

    train: list[dict] = []
    for s in skills:
        area = set(s["frameworks"]) or set(s["languages"])
        field = "frameworks" if s["frameworks"] else "languages"
        mates = sorted(o["name"] for o in skills if o["name"] != s["name"]
                       and area & set(o[field]))
        rng.shuffle(mates)
        names = [s["name"]] + mates[:HARD_NEGATIVES]
        rng.shuffle(names)
        keys = names + ["none"]
        texts = [by_name[n]["description"] for n in names] + [clm.NONE_OPTION]
        for kind, gold_key in (("should", s["name"]), ("should_not", "none")):
            for i, case in enumerate(s["tests"].get(kind) or []):
                text = (case or {}).get("text") or ""
                if not text.strip():
                    continue
                if leaks(text):
                    stats["train_dropped_overlap"] += 1
                    continue
                train.append(_row(f"act:{s['name']}:{kind}:{i}", text, keys,
                                  texts, {gold_key: 1.0}, f"activation_{kind}"))
                stats[f"train_activation_{kind}"] += 1
    # Client traffic only.
    groups = defaultdict(list)
    if os.path.exists(LABELS):
        with open(LABELS, encoding="utf-8") as fh:
            for ln in fh:
                if not ln.strip():
                    continue
                d = json.loads(ln)
                if d.get("traffic") != "client":
                    stats["labels_not_client_skipped"] += 1
                    continue
                groups[d.get("fallback")].append(d)
    for fb, rows in sorted(groups.items()):
        q = rows[0].get("query") or ""
        cands = [ids.get(r.get("skill")) for r in rows]
        if not q.strip() or not any(cands):
            stats["client_unusable"] += 1
            continue
        if leaks(q):
            stats["train_dropped_overlap"] += 1
            continue
        names = list(dict.fromkeys(n for n in cands if n))
        used = [ids.get(r["skill"]) for r in rows
                if str(r.get("label")) == "1" and ids.get(r.get("skill"))]
        gold = ({n: 1.0 / len(used) for n in used} if used else {"none": 1.0})
        keys = names + ["none"]
        texts = [by_name[n]["description"] for n in names] + [clm.NONE_OPTION]
        train.append(_row(f"client:{fb}", q, keys, texts, gold, "client"))
        stats["train_client"] += 1
    # The daily eval: the held-out test split.
    test: list[dict] = []
    for r in fidelity._daily_rows():
        if not r.get("user") or r.get("source"):
            continue
        must = [m for m in r.get("must") or [] if m in by_name]
        nothing = bool(r.get("nothing"))
        if not must and not nothing:
            stats["test_area_only_skipped"] += 1
            continue
        lex = [n for _, n in fidelity._lexical(r["user"], skills)
               if n not in must]
        names = (must + lex)[:fidelity.N_SKILL_OPTIONS]
        keys = names + ["none"]
        texts = [by_name[n]["description"] for n in names] + [clm.NONE_OPTION]
        gold = {must[0]: 1.0} if must else {"none": 1.0}
        test.append(_row(f"daily:{r['id']}", r["user"], keys, texts, gold,
                         "daily_eval"))
        stats["test_daily"] += 1
    import pyarrow as pa
    import pyarrow.parquet as pq
    os.makedirs(os.path.join(DATA, WORKFLOW), exist_ok=True)
    files = {}
    for split, rows in (("train", train), ("test", test)):
        p = os.path.join(DATA, WORKFLOW, f"{split}-00000-of-00001.parquet")
        cols = {k: [r[k] for r in rows] for k in
                ("id", "workflow", "source", "state", "questions", "gold")}
        pq.write_table(pa.table(cols), p, compression="zstd")
        with open(p, "rb") as fh:
            files[os.path.relpath(p, OUT).replace(os.sep, "/")] = {
                "rows": len(rows), "sha256": hashlib.sha256(fh.read()).hexdigest()}
    texts = all_texts(train + test)
    man = {"stats": dict(stats), "files": files, "texts": len(texts),
           "workflow": WORKFLOW, "seed": SEED, "instructions": clm.INSTRUCTIONS,
           "none_option": clm.NONE_OPTION, "hard_negatives": HARD_NEGATIVES,
           "shingle_words": SHINGLE}
    with open(os.path.join(OUT, "MANIFEST.json"), "w", encoding="utf-8",
              newline="\n") as fh:
        json.dump(man, fh, indent=1, sort_keys=True)
    return man


def all_texts(rows: list[dict]) -> list[str]:
    """Every state text and candidate text finetune.py will look up, exactly
    as clm.schema.build_pairs builds them."""
    out = []
    for r in rows:
        for q in json.loads(r["questions"]).values():
            out.append(clm.state_text(r["state"], q["instructions"]))
            out.extend(q["criteria"].values())
    return list(dict.fromkeys(out))


def _read_rows() -> list[dict]:
    import pyarrow.parquet as pq
    rows = []
    for split in ("train", "test"):
        p = os.path.join(DATA, WORKFLOW, f"{split}-00000-of-00001.parquet")
        rows += pq.read_table(p).to_pylist()
    return rows


def embed(url: str) -> dict:
    """Fill finetune.py's TextCache from OUR encoder at `url`."""
    texts = all_texts(_read_rows())
    enc = clm.HttpEncoder(url=url.rstrip("/") + "/v1/embeddings", model=None,
                          gpu_room_model=None)
    path = cache_path()
    have = {}
    if os.path.exists(path):
        z = np.load(path)
        have = dict(zip(z["keys"].tolist(), z["vecs"]))
    todo = [t for t in texts if hashlib.sha1(t.encode()).hexdigest() not in have]
    import time
    t0 = time.time()
    for i in range(0, len(todo), 64):
        chunk = todo[i:i + 64]
        v = enc.embed_texts(chunk)
        for t, row in zip(chunk, v):
            have[hashlib.sha1(t.encode()).hexdigest()] = row.astype(np.float16)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ks = list(have)
    np.savez(path, keys=np.array(ks), vecs=np.stack([have[k] for k in ks])
             .astype(np.float16))
    r = {"texts": len(texts), "embedded": len(todo), "cache": path,
         "seconds": round(time.time() - t0, 1), "encoder": clm.ENCODER_ID}
    with open(os.path.join(OUT, "EMBEDDINGS.json"), "w", encoding="utf-8",
              newline="\n") as fh:
        json.dump(r, fh, indent=1, sort_keys=True)
    return r


def command() -> str:
    pt = os.path.join(MODELS_DIR, "CLM-v0.1-8B", "CLM_v0.1-8B.pt")
    return (f"git clone {CLM_REPO} CLM && git -C CLM checkout {CLM_COMMIT}\n"
            f"python CLM/train/finetune.py --task choice "
            f"--data {DATA} --workflow {WORKFLOW} "
            f"--init-ckpt {pt} "
            f"--embed-cache {os.path.dirname(cache_path())} "
            f"--embed-model {EMBED_MODEL} --max-len {MAX_LEN} "
            f"--loss infonce --targets soft --batch 256 --epochs 20 "
            f"--patience 5 --val-frac 0.1 --seed 1234 "
            f"--out-dir {os.path.join(OUT, 'runs', 'skills-v1')}")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    e = sub.add_parser("embed")
    e.add_argument("--url", required=True)
    sub.add_parser("command")
    a = ap.parse_args(argv)
    if a.cmd == "build":
        print(json.dumps(build(), indent=1))
    elif a.cmd == "embed":
        print(json.dumps(embed(a.url), indent=1))
    else:
        print(command())
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
