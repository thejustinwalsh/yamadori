#!/usr/bin/env python
"""CLM encoder fidelity: does a quantized llama.cpp Qwen3-8B feed the CLM heads
what the official path feeds them? (docs/CLM.md "Fidelity"; the G1/G2 gates
of docs/CLM-EVAL.md section 5, stage 0.)

    python bench/clm/fidelity.py build                 # the question set
    python bench/clm/fidelity.py selftest              # layer-streamed == Qwen3Model (tiny random config)
    python bench/clm/fidelity.py reference [--threads 6]   # HF transformers, fp32, CPU
    python bench/clm/fidelity.py served --tag q8_0 --url http://127.0.0.1:PORT
    python bench/clm/fidelity.py compare               # the table

THE QUESTIONS (bench/clm/fidelity_questions.jsonl, written by `build`,
deterministic): 40 (state, options) skill-selection questions --
  * 22 user requests from bench/skills/daily_eval.jsonl (the rows with a
    `user` text; `source` rows read test material and are skipped), two per
    category where the category has two, seeded;
  *  4 agent-step states from its `turns` sequences (the user turn followed by
    the tool results, the caller's evidence order);
  * 14 from the skills' activation tests (index/skills/library/*/tests.json):
    7 `should` cases and 7 `should_not` near misses (the first should_not,
    which skill_tests puts first), each against its own skill plus 4 skills
    of the same framework or language.
Options are skill trigger texts (the SKILL.md `description`) plus one "none"
option. Daily-eval options are the skills its `must` names plus the top
skills by a lexical (IDF-weighted word overlap) score, 6 skills in all. The
question is CLM's layout: `state + "\\n\\n" + INSTRUCTIONS` for the state
head, each option's text alone for the action head (src/clm/schema.py).

THE REFERENCE (`reference`): the official encode path is vLLM's pooling
runner on Qwen/Qwen3-8B -- the final hidden state (after the model's last
RMSNorm) of the LAST token, L2-normalised (src/clm/embedder.py,
train/embed_utils.py), tokenised with add_special_tokens=False and the tail
kept (Recipe.text_ids(keep="tail"), cap max_len - 1 = 2047). This run
computes the same with HF transformers' own Qwen3 modules (Qwen3DecoderLayer,
Qwen3RotaryEmbedding, create_causal_mask, Qwen3RMSNorm) in float32 on the CPU,
streaming ONE LAYER'S WEIGHTS AT A TIME from the pinned safetensors (all
texts pass through layer i before layer i+1 is read): the 16.4 GB model does
not fit this host's free commit next to the running stack. `selftest` checks
the streaming against Qwen3Model.forward on a tiny random config.

THE SERVED PATH (`served`): mcp/clm.py's own encoder request -- token ids
from the pinned tokenizer.json, POST /v1/embeddings -- against a llama-server
running config.yaml's `clm-encoder` flags with the given GGUF. Also G2: the
server's /tokenize of every text vs the local ids.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "mcp"))

QUESTIONS = os.path.join(HERE, "fidelity_questions.jsonl")
RESULTS = os.path.join(HERE, "results")
VECTORS = os.path.join(ROOT, "index", "clm", "fidelity")
DAILY = os.path.join(ROOT, "bench", "skills", "daily_eval.jsonl")
LIBRARY = os.path.join(ROOT, "index", "skills", "library")
MODEL_DIR = os.environ.get("YAMADORI_CLM_HF_DIR",
                           "C:/Users/jwals/textgen/user_data/models/Qwen3-8B")

SEED = 20260927
N_SKILL_OPTIONS = 6

import clm  # noqa: E402  (INSTRUCTIONS, NONE_OPTION, state_text, tokenizer)


def key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


# ------------------------------------------------------------------ build --
_WORD = re.compile(r"[a-z][a-z0-9_+#.-]{2,}")


def _words(t: str) -> list[str]:
    return [w.strip(".-") for w in _WORD.findall((t or "").lower())]


def _skills() -> list[dict]:
    import skill_md
    out = []
    for folder in skill_md.folders(LIBRARY):
        sk, tests = skill_md.read_folder(folder)
        y = sk.get("yamadori") or {}
        if y.get("state") not in (None, "armed") or not sk.get("description"):
            continue
        cat = y.get("category") or {}
        aw = y.get("applies_when") or {}
        out.append({"name": sk["name"], "description": sk["description"].strip(),
                    "revision": y.get("revision"),
                    "frameworks": sorted(cat.get("framework") or []),
                    "languages": sorted(cat.get("language") or []),
                    "bag": " ".join([sk["description"], " ".join(sk.get("tags") or []),
                                     aw.get("text") or ""]),
                    "tests": (tests or {}).get("activation") or {}})
    return sorted(out, key=lambda s: s["name"])


def _lexical(state: str, skills: list[dict]) -> list[tuple[float, str]]:
    df = Counter()
    bags = {}
    for s in skills:
        ws = set(_words(s["bag"]))
        bags[s["name"]] = ws
        df.update(ws)
    n = len(skills)
    q = set(_words(state))
    scored = []
    for s in skills:
        sc = sum(math.log(1 + n / df[w]) for w in q & bags[s["name"]])
        scored.append((sc, s["name"]))
    return sorted(scored, key=lambda t: (-t[0], t[1]))


def _question(qid, source, state, skill_names, by_name, expect=None):
    keys = list(skill_names) + ["none"]
    texts = [by_name[n]["description"] for n in skill_names] + [clm.NONE_OPTION]
    return {"id": qid, "source": source, "state": state,
            "instructions": clm.INSTRUCTIONS, "keys": keys, "options": texts,
            "expect": expect}


def _daily_rows() -> list[dict]:
    rows = []
    with open(DAILY, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("//"):
                rows.append(json.loads(line))
    return rows


def _sequence_state(turns: list[dict]) -> str:
    """The user turn, then each tool step as `name(args) -> result`: the
    caller's evidence, in its order."""
    parts = []
    for t in turns:
        if "user" in t:
            parts.append(t["user"])
        elif "step" in t:
            st = t["step"]
            args = json.dumps(st.get("args") or {}, sort_keys=True)
            parts.append(f"{st.get('name')}({args}) ->\n{st.get('result', '')}")
    return "\n\n".join(parts)


def build() -> list[dict]:
    rng = random.Random(SEED)
    skills = _skills()
    by_name = {s["name"]: s for s in skills}
    qs: list[dict] = []
    daily = _daily_rows()
    # 22 user rows: two per category (one where only one exists), seeded.
    by_cat = defaultdict(list)
    for r in daily:
        if r.get("user") and not r.get("source"):
            by_cat[r["cat"]].append(r)
    picked = []
    for cat in sorted(by_cat):
        rows = sorted(by_cat[cat], key=lambda r: r["id"])
        rng.shuffle(rows)
        picked += rows[:2]
    extra = sorted((r for c in by_cat.values() for r in c if r not in picked),
                   key=lambda r: r["id"])
    rng.shuffle(extra)
    picked = (picked + extra)[:22]
    for r in sorted(picked, key=lambda r: r["id"]):
        must = [m for m in r.get("must") or [] if m in by_name]
        lex = [n for _, n in _lexical(r["user"], skills) if n not in must]
        names = (must + lex)[:N_SKILL_OPTIONS]
        qs.append(_question(f"daily:{r['id']}", "daily_eval", r["user"], names,
                            by_name, {"must": r.get("must"),
                                      "nothing": r.get("nothing")}))
    # 4 agent-step sequences, the whole evidence as the state.
    seqs = sorted((r for r in daily if r.get("turns")), key=lambda r: r["id"])
    rng.shuffle(seqs)
    for r in sorted(seqs[:4], key=lambda r: r["id"]):
        state = _sequence_state(r["turns"])
        lex = [n for _, n in _lexical(state, skills)]
        qs.append(_question(f"seq:{r['id']}", "daily_eval_turns", state,
                            lex[:N_SKILL_OPTIONS], by_name, None))
    # 14 activation cases: 7 skills, each with a should and its near miss.
    cands = [s for s in skills if s["tests"].get("should")
             and s["tests"].get("should_not") and (s["frameworks"] or s["languages"])]
    rng.shuffle(cands)
    used = 0
    for s in cands:
        if used == 7:
            break
        area = set(s["frameworks"]) or set(s["languages"])
        mates = [o["name"] for o in skills if o["name"] != s["name"]
                 and (area & set(o["frameworks"] if s["frameworks"] else o["languages"]))]
        if len(mates) < 4:
            continue
        rng.shuffle(mates)
        names = [s["name"]] + sorted(mates[:4])
        rng.shuffle(names)
        should = s["tests"]["should"][0]["text"]
        near = s["tests"]["should_not"][0]["text"]
        qs.append(_question(f"act:{s['name']}:should", "activation_should",
                            should, names, by_name, {"skill": s["name"]}))
        qs.append(_question(f"act:{s['name']}:near_miss", "activation_near_miss",
                            near, names, by_name, {"skill": None}))
        used += 1
    with open(QUESTIONS, "w", encoding="utf-8", newline="\n") as fh:
        for q in qs:
            fh.write(json.dumps(q, ensure_ascii=False, sort_keys=True) + "\n")
    return qs


def load_questions() -> list[dict]:
    with open(QUESTIONS, encoding="utf-8") as fh:
        return [json.loads(x) for x in fh if x.strip()]


def texts_of(qs: list[dict]) -> tuple[list[str], list[str]]:
    """(state texts, option texts), each unique, in first-seen order."""
    st = list(dict.fromkeys(clm.state_text(q["state"], q["instructions"])
                            for q in qs))
    op = list(dict.fromkeys(t for q in qs for t in q["options"]))
    return st, op


# ------------------------------------------------------ the HF reference --
def _stream_forward(id_lists, config, tensor, n_layers, threads=6, log=print):
    """Qwen3Model.forward, one layer's weights resident at a time, float32.
    `tensor(name)` returns a checkpoint tensor. Returns [n, hidden] final
    hidden states (after model.norm) of each sequence's LAST token."""
    import torch
    from transformers.masking_utils import create_causal_mask
    from transformers.models.qwen3.modeling_qwen3 import (
        Qwen3DecoderLayer, Qwen3RMSNorm, Qwen3RotaryEmbedding)
    torch.set_num_threads(threads)
    with torch.no_grad():
        emb = tensor("model.embed_tokens.weight")
        hs = [emb[torch.tensor(ids, dtype=torch.long)].float().unsqueeze(0)
              for ids in id_lists]
        del emb
        rotary = Qwen3RotaryEmbedding(config)
        pos = [torch.arange(h.shape[1]).unsqueeze(0) for h in hs]
        pe = [rotary(h, p) for h, p in zip(hs, pos)]
        masks = [create_causal_mask(config=config, inputs_embeds=h,
                                    attention_mask=None, past_key_values=None,
                                    position_ids=p) for h, p in zip(hs, pos)]
        for i in range(n_layers):
            t0 = time.time()
            with torch.device("meta"):
                layer = Qwen3DecoderLayer(config, i)
            pre = f"model.layers.{i}."
            sd = {n: tensor(pre + n).float()
                  for n in layer.state_dict().keys()}
            layer.load_state_dict(sd, strict=True, assign=True)
            layer.eval()
            for j in range(len(hs)):
                hs[j] = layer(hs[j], attention_mask=masks[j],
                              position_embeddings=pe[j], position_ids=pos[j])
            del layer, sd
            log(f"  layer {i + 1}/{n_layers} {time.time() - t0:.1f}s")
        norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        norm.load_state_dict({"weight": tensor("model.norm.weight").float()})
        return np.stack([norm(h)[0, -1].numpy().astype(np.float32) for h in hs])


def selftest() -> dict:
    """The streamed forward == Qwen3Model.forward on a tiny random config."""
    import torch
    from transformers import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Model
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=97, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=3, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=16, rms_norm_eps=1e-6,
                      rope_theta=1_000_000, max_position_embeddings=256)
    cfg._attn_implementation = "sdpa"
    m = Qwen3Model(cfg).eval()
    sd = {f"model.{k}": v for k, v in m.state_dict().items()}
    ids = [[1, 5, 9, 33, 2], [7, 7, 8], list(range(40))]
    with torch.no_grad():
        ref = np.stack([m(input_ids=torch.tensor([x])).last_hidden_state[0, -1]
                        .numpy() for x in ids])
    got = _stream_forward(ids, cfg, lambda n: sd[n].clone(), 3, threads=1,
                          log=lambda *_: None)
    d = float(np.abs(ref - got).max())
    return {"max_abs": d, "ok": d < 1e-5}


def reference(threads: int = 6) -> dict:
    import torch  # noqa: F401
    from safetensors import safe_open
    from transformers import AutoConfig, AutoTokenizer
    qs = load_questions()
    st, op = texts_of(qs)
    texts = st + op
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    cap = clm.MAX_TOKENS - 1
    ids = []
    for t in texts:
        x = tok(t, add_special_tokens=False)["input_ids"]
        ids.append((x or tok(" ", add_special_tokens=False)["input_ids"])[-cap:])
    config = AutoConfig.from_pretrained(MODEL_DIR)
    config._attn_implementation = "sdpa"
    with open(os.path.join(MODEL_DIR, "model.safetensors.index.json")) as fh:
        wmap = json.load(fh)["weight_map"]
    handles = {}

    def tensor(name):
        f = wmap[name]
        if f not in handles:
            handles[f] = safe_open(os.path.join(MODEL_DIR, f), "pt")
        return handles[f].get_tensor(name)

    t0 = time.time()
    print(f"reference: {len(texts)} texts, {sum(map(len, ids))} tokens, "
          f"{threads} threads", flush=True)
    vec = _stream_forward(ids, config, tensor, config.num_hidden_layers,
                          threads=threads)
    os.makedirs(VECTORS, exist_ok=True)
    out = os.path.join(VECTORS, "reference_fp32.npz")
    np.savez(out, keys=np.array([key(t) for t in texts]), vecs=vec,
             n_tokens=np.array([len(x) for x in ids]),
             ids_sha=np.array([hashlib.sha1(json.dumps(x).encode()).hexdigest()
                               for x in ids]))
    r = {"texts": len(texts), "tokens": int(sum(map(len, ids))),
         "seconds": round(time.time() - t0, 1), "out": out, "threads": threads}
    print(json.dumps(r), flush=True)
    return r


# --------------------------------------------------------- the served path --
def served(tag: str, url: str) -> dict:
    """Embed every text through mcp/clm.py's encoder request against `url`
    (a llama-server with the clm-encoder flags) and check /tokenize (G2)."""
    import urllib.request
    qs = load_questions()
    st, op = texts_of(qs)
    texts = st + op
    enc = clm.HttpEncoder(url=url.rstrip("/") + "/v1/embeddings", model=None,
                          gpu_room_model=None)
    t0 = time.time()
    vec = enc.embed_texts(texts)
    ms = (time.time() - t0) * 1000
    # G2: the server's own tokenisation of the text vs the ids we send.
    mism = []
    tk = clm.tokenizer()
    for t in texts:
        body = json.dumps({"content": t, "add_special": True,
                           "parse_special": True}).encode()
        req = urllib.request.Request(url.rstrip("/") + "/tokenize", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            srv = json.loads(r.read())["tokens"]
        mine = tk.ids(t)
        if srv != mine:
            mism.append({"key": key(t)[:10], "server": len(srv), "local": len(mine)})
    os.makedirs(VECTORS, exist_ok=True)
    out = os.path.join(VECTORS, f"served_{tag}.npz")
    np.savez(out, keys=np.array([key(t) for t in texts]), vecs=vec)
    r = {"tag": tag, "texts": len(texts), "ms_total": round(ms),
         "tokenize_mismatch": mism, "out": out}
    print(json.dumps(r), flush=True)
    return r


# --------------------------------------------------------------- compare --
def _load(path):
    z = np.load(path)
    return {k: v for k, v in zip(z["keys"].tolist(), z["vecs"])}


def compare(tags: list[str]) -> dict:
    import clm_heads
    heads = clm_heads.load()
    qs = load_questions()
    ref = _load(os.path.join(VECTORS, "reference_fp32.npz"))
    arms = {t: _load(os.path.join(VECTORS, f"served_{t}.npz")) for t in tags
            if os.path.exists(os.path.join(VECTORS, f"served_{t}.npz"))}

    def decide(vecs, q):
        s = vecs[key(clm.state_text(q["state"], q["instructions"]))]
        a = np.stack([vecs[key(t)] for t in q["options"]])
        zs = heads.project_states(s)[0]
        za = heads.project_actions(a)
        return clm_heads.softmax(heads.logits(zs, za))

    refp = {q["id"]: decide(ref, q) for q in qs}
    table = {}
    st, op = texts_of(qs)
    for tag, v in arms.items():
        cs = [float(clm_heads.l2(v[key(t)]) @ clm_heads.l2(ref[key(t)]))
              for t in st]
        co = [float(clm_heads.l2(v[key(t)]) @ clm_heads.l2(ref[key(t)]))
              for t in op]
        agree = none_agree = 0
        dp, rows = [], []
        for q in qs:
            p = decide(v, q)
            r = refp[q["id"]]
            none_i = q["keys"].index("none")
            a_same = int(np.argmax(p)) == int(np.argmax(r))
            n_same = (int(np.argmax(p)) == none_i) == (int(np.argmax(r)) == none_i)
            agree += a_same
            none_agree += n_same
            dp.append(float(np.abs(p - r).max()))
            if not a_same or not n_same:
                rows.append({"id": q["id"], "ref": q["keys"][int(np.argmax(r))],
                             "ref_p": round(float(r.max()), 3),
                             "got": q["keys"][int(np.argmax(p))],
                             "got_p": round(float(p.max()), 3)})
        table[tag] = {
            "states": len(st), "options": len(op),
            "cos_state_median": round(float(np.median(cs)), 5),
            "cos_state_min": round(float(np.min(cs)), 5),
            "cos_option_median": round(float(np.median(co)), 5),
            "cos_option_min": round(float(np.min(co)), 5),
            "argmax_agree": f"{agree}/{len(qs)}",
            "none_vs_pick_agree": f"{none_agree}/{len(qs)}",
            "max_abs_prob_diff_median": round(float(np.median(dp)), 4),
            "max_abs_prob_diff_max": round(float(np.max(dp)), 4),
            "disagreements": rows}
    ref_summary = {
        "none_picked": sum(int(np.argmax(refp[q["id"]])) == q["keys"].index("none")
                           for q in qs),
        "top_p_median": round(float(np.median([p.max() for p in refp.values()])), 3)}
    out = {"questions": len(qs), "reference": ref_summary, "arms": table}
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "fidelity.json"), "w", encoding="utf-8",
              newline="\n") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    sub.add_parser("selftest")
    r = sub.add_parser("reference")
    r.add_argument("--threads", type=int, default=6)
    s = sub.add_parser("served")
    s.add_argument("--tag", required=True)
    s.add_argument("--url", required=True)
    c = sub.add_parser("compare")
    c.add_argument("--tags", default="bf16_cpu,q8_0_cpu,q6_k_cpu,q8_0,q6_k")
    a = ap.parse_args(argv)
    if a.cmd == "build":
        qs = build()
        st, op = texts_of(qs)
        print(json.dumps({"questions": len(qs), "states": len(st),
                          "options": len(op),
                          "by_source": Counter(q["source"] for q in qs)}))
    elif a.cmd == "selftest":
        r = selftest()
        print(json.dumps(r))
        return 0 if r["ok"] else 1
    elif a.cmd == "reference":
        reference(a.threads)
    elif a.cmd == "served":
        served(a.tag, a.url)
    else:
        print(json.dumps(compare(a.tags.split(",")), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
