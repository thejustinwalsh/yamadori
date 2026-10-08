#!/usr/bin/env python
"""Clef on the Bonsai engine: converter -> GGUF -> libllama, against a PyTorch reference. CPU only, no GPU, no network.

What it does (docs/ENGINES.md "Clef in the Bonsai fork"):

  1. builds a TINY Clef checkpoint: a random Qwen3.5 text model saved by transformers (config.json, safetensors), the
     tokenizer files of a local Qwen3.5-family model (copied, not downloaded), and a random joint head in the layout
     the converter reads (joint_head_config.json, joint_head.safetensors; the tensor names are the ones
     conversion/clef.py and gguf-py's tensor map expect)
  2. converts it with the engine tree's own conversion/clef.py (the converter this fork ships)
  3. computes the scores of a request-shaped prompt (a few spans of question and option tokens) in PyTorch: the
     transformers Qwen3.5 trunk (the real model code, not ours) and a straightforward float32 version of the head
  4. runs the engine's test-clef in dump mode on the GGUF and compares the scores

The head reference is written from the graph of ggml-org PR 29831 (src/models/clef.cpp, "ref: JointSchemaHead"), because
Cloudflare's own joint_schema_model.py is not in this tree: what this proves is that the converter, the loader, the
batch plumbing, the no-memory Qwen3.5 trunk and the head graph compute what that description says, on the CPU. That the
description is Cloudflare's model is for a trained checkpoint to show (the GPU gate plan, docs/ENGINES.md).

    python bench/clef_smoke.py --engine-src engines/src/llama-bonsai2-ada --test-clef <build>/bin/test-clef.exe \\
        --tokenizer-from C:/Users/jwals/textgen/user_data/models/Qwen3-8B
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

N_EMBD = 64
N_EMBD_H = 32
N_FF_H = 64
N_HEAD_DEC = 2
ROUTING_LAYERS = 1
JOINT_LAYERS = 1

NONE, Q_NOUL, Q_CHOICE, Q_SCORE, OPTION = 0, 1, 2, 3, 4


def make_trunk(out: Path, tokenizer_from: Path, seed: int) -> dict:
    import torch
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    tok = json.loads((tokenizer_from / "tokenizer.json").read_text(encoding="utf-8"))
    vocab = len(tok["model"]["vocab"]) + len(tok.get("added_tokens", []))
    cfg = Qwen3_5TextConfig(
        vocab_size=vocab + 8, hidden_size=N_EMBD, intermediate_size=128, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16, linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        rms_norm_eps=1e-6, tie_word_embeddings=False, max_position_embeddings=512,
        rope_parameters={"mrope_interleaved": True, "mrope_section": [2, 1, 1], "partial_rotary_factor": 0.5,
                         "rope_theta": 10000.0, "rope_type": "default"},
        partial_rotary_factor=0.5, hidden_act="silu",
    )
    torch.manual_seed(seed)
    model = Qwen3_5ForCausalLM(cfg).float().eval()
    # the default init leaves the norms near 0 / identity and the gates flat: widen them so every layer matters
    with torch.no_grad():
        for n, p in model.named_parameters():
            if p.ndim >= 2:
                p.normal_(0.0, 0.08)
    model.save_pretrained(out)
    for f in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"):
        if (tokenizer_from / f).exists():
            shutil.copy(tokenizer_from / f, out / f)
    return {"vocab": cfg.vocab_size}


def make_head(out: Path, n_embd: int, seed: int) -> dict:
    import torch
    from safetensors.torch import save_file

    g = torch.Generator().manual_seed(seed)
    h, ff = N_EMBD_H, N_FF_H

    def rnd(*shape, std=0.1):
        return torch.randn(*shape, generator=g) * std

    def norm(prefix, n):
        return {prefix + ".weight": 1.0 + rnd(n, std=0.05), prefix + ".bias": rnd(n, std=0.05)}

    def mha(prefix):
        return {prefix + ".in_proj_weight": rnd(3 * h, h), prefix + ".in_proj_bias": rnd(3 * h, std=0.05),
                prefix + ".out_proj.weight": rnd(h, h), prefix + ".out_proj.bias": rnd(h, std=0.05)}

    def lin(prefix, n_out, n_in, bias=True):
        d = {prefix + ".weight": rnd(n_out, n_in)}
        if bias:
            d[prefix + ".bias"] = rnd(n_out, std=0.05)
        return d

    t: dict = {}
    t["type_embedding.weight"] = rnd(3, h)  # nn.Embedding(3, h)
    for i in range(ROUTING_LAYERS):
        p = f"evidence_layers.{i}"
        t.update(norm(p + ".query_norm", h)); t.update(norm(p + ".memory_norm", h))
        t.update(mha(p + ".attention"))
        t.update(norm(p + ".feedforward_norm", h))
        t.update(lin(p + ".feedforward.0", ff, h)); t.update(lin(p + ".feedforward.3", h, ff))
    for i in range(JOINT_LAYERS):
        p = f"layers.{i}"
        t.update(norm(p + ".norm1", h)); t.update(mha(p + ".self_attn"))
        t.update(norm(p + ".norm2", h)); t.update(mha(p + ".multihead_attn"))
        t.update(norm(p + ".norm3", h))
        t.update(lin(p + ".linear1", ff, h)); t.update(lin(p + ".linear2", h, ff))
    t.update(norm("hidden_norm", n_embd))
    for name in ("memory_projection", "question_projection", "option_question_projection", "global_projection",
                 "option_context_projection", "option_lexical_projection"):
        t.update(lin(name, h, n_embd, bias=False))
    t.update(norm("option_summary_norm", h)); t.update(norm("field_norm", h)); t.update(norm("option_norm", h))
    t.update(lin("residual_scorer.0", h, 4 * h)); t.update(lin("residual_scorer.3", 1, h))
    t["prior_logit_scale"] = torch.tensor(1.1); t["joint_logit_scale"] = torch.tensor(1.6); t["residual_gate"] = torch.tensor(0.4)
    save_file({k: v.contiguous() for k, v in t.items()}, str(out / "joint_head.safetensors"))
    (out / "joint_head_config.json").write_text(json.dumps(
        {"routing_layers": ROUTING_LAYERS, "layers": JOINT_LAYERS, "heads": N_HEAD_DEC}), encoding="utf-8")
    return t


def head_reference(hidden, out_weight, tokens, orders, t):
    """JointSchemaHead, in float32. hidden: [n_tokens, n_embd] after the trunk's final norm."""
    import torch
    import torch.nn.functional as F

    h = N_EMBD_H
    nh = N_HEAD_DEC

    def ln(x, p):
        return F.layer_norm(x, (x.shape[-1],), t[p + ".weight"], t[p + ".bias"], 1e-5)

    def mha(q_in, kv_in, p):
        w, b = t[p + ".in_proj_weight"], t[p + ".in_proj_bias"]
        q = q_in @ w[:h].T + b[:h]
        k = kv_in @ w[h:2 * h].T + b[h:2 * h]
        v = kv_in @ w[2 * h:].T + b[2 * h:]
        d = h // nh
        qh = q.view(-1, nh, d).transpose(0, 1)
        kh = k.view(-1, nh, d).transpose(0, 1)
        vh = v.view(-1, nh, d).transpose(0, 1)
        a = torch.softmax(qh @ kh.transpose(1, 2) / math.sqrt(d), dim=-1)
        o = (a @ vh).transpose(0, 1).reshape(-1, h)
        return o @ t[p + ".out_proj.weight"].T + t[p + ".out_proj.bias"]

    def ffn(x, pn, p1, p2):
        x = ln(x, pn)
        x = F.gelu(x @ t[p1 + ".weight"].T + t[p1 + ".bias"])
        return x @ t[p2 + ".weight"].T + t[p2 + ".bias"]

    # spans
    questions, options = [], []
    i, n = 0, len(orders)
    while i < n:
        o = orders[i]
        j = i + 1
        while j < n and orders[j] == o:
            j += 1
        if o in (Q_NOUL, Q_CHOICE, Q_SCORE):
            questions.append((o - Q_NOUL, i, j))
        elif o == OPTION:
            options.append((len(questions) - 1, i, j))
        i = j
    hid = ln(hidden, "hidden_norm")
    memory = hid @ t["memory_projection.weight"].T
    glob = hid[-1]
    question = torch.stack([hid[a:b].mean(0) for _, a, b in questions])
    opt_ctx = torch.stack([hid[a:b].mean(0) for _, a, b in options])
    lexical = torch.stack([out_weight[torch.tensor(tokens[a:b])].mean(0) for _, a, b in options])
    owner = torch.tensor([q for q, _, _ in options])
    qtype = torch.tensor([ty for ty, _, _ in questions])

    opts = (opt_ctx @ t["option_context_projection.weight"].T + lexical @ t["option_lexical_projection.weight"].T
            + (question @ t["option_question_projection.weight"].T)[owner])
    for l in range(ROUTING_LAYERS):
        p = f"evidence_layers.{l}"
        opts = opts + mha(ln(opts, p + ".query_norm"), ln(memory, p + ".memory_norm"), p + ".attention")
        opts = opts + ffn(opts, p + ".feedforward_norm", p + ".feedforward.0", p + ".feedforward.3")

    fields = question @ t["question_projection.weight"].T
    mask = torch.full((len(questions), len(options)), -float("inf"))
    for oi, (qi, _, _) in enumerate(options):
        mask[qi, oi] = 0.0
    w = torch.softmax(fields @ opts.T / math.sqrt(h) + mask, dim=-1)
    fields = fields + ln(w @ opts, "option_summary_norm") + (glob @ t["global_projection.weight"].T) + t["type_embedding.weight"][qtype]
    for l in range(JOINT_LAYERS):
        p = f"layers.{l}"
        c = ln(fields, p + ".norm1")
        fields = fields + mha(c, c, p + ".self_attn")
        c = ln(fields, p + ".norm2")
        fields = fields + mha(c, memory, p + ".multihead_attn")
        fields = fields + ffn(fields, p + ".norm3", p + ".linear1", p + ".linear2")
    fields = ln(fields, "field_norm")

    def l2(x):
        return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    anchor = l2(question + glob)[owner]
    prior = (l2(lexical) * anchor).sum(-1)
    opts_n = ln(opts, "option_norm")
    field = fields[owner]
    cosine = (l2(field) * l2(opts_n)).sum(-1)
    feats = torch.cat([field, opts_n, field * opts_n, (field - opts_n).abs()], dim=-1)
    resid = F.gelu(feats @ t["residual_scorer.0.weight"].T + t["residual_scorer.0.bias"]) @ t["residual_scorer.3.weight"].T + t["residual_scorer.3.bias"]
    s0 = math.exp(min(float(t["prior_logit_scale"]), math.log(100.0)))
    s1 = math.exp(min(float(t["joint_logit_scale"]), math.log(100.0)))
    s2 = 1.0 / (1.0 + math.exp(-float(t["residual_gate"])))
    joint = cosine * s1 + resid.squeeze(-1)
    return prior * s0 + joint * s2


def make_prompt(rng: random.Random, vocab: int):
    toks, orders = [], []

    def run(order, n):
        for _ in range(n):
            toks.append(rng.randrange(3, vocab - 8))
            orders.append(order)

    run(NONE, 9); run(Q_NOUL, 5); run(NONE, 2); run(OPTION, 3); run(NONE, 1); run(OPTION, 2); run(NONE, 3)
    run(Q_CHOICE, 4); run(NONE, 2); run(OPTION, 2); run(NONE, 1); run(OPTION, 4); run(NONE, 1); run(OPTION, 1); run(NONE, 6)
    return toks, orders


def main() -> int:
    # fla's Triton kernels (when installed) cannot take CPU tensors: make transformers use its own torch implementation
    import transformers.utils.import_utils as iu
    iu.is_flash_linear_attention_available = lambda *a, **k: False
    iu.is_causal_conv1d_available = lambda *a, **k: False

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--engine-src", required=True, help="the engine tree (engines/src/llama-bonsai2-ada): conversion/ and gguf-py/")
    ap.add_argument("--test-clef", required=True, help="the engine's test-clef executable (CPU)")
    ap.add_argument("--tokenizer-from", required=True, help="a local directory with the tokenizer files of a Qwen3.5-family model")
    ap.add_argument("--work", default=None, help="working directory (default: a temp dir, removed)")
    ap.add_argument("--tol", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=20261008)
    args = ap.parse_args()

    import numpy as np  # noqa: F401
    import torch
    from safetensors.torch import load_file

    src = Path(args.engine_src).resolve()
    work = Path(args.work) if args.work else Path(tempfile.mkdtemp(prefix="clef-smoke-"))
    work.mkdir(parents=True, exist_ok=True)
    ck = work / "checkpoint"
    ck.mkdir(exist_ok=True)
    try:
        info = make_trunk(ck, Path(args.tokenizer_from), args.seed)
        head = make_head(ck, N_EMBD, args.seed + 1)

        gguf_path = work / "tiny-clef.gguf"
        env = dict(os.environ, PYTHONPATH=str(src / "gguf-py"))
        cmd = [sys.executable, str(src / "convert_hf_to_gguf.py"), str(ck), "--outfile", str(gguf_path), "--outtype", "f32"]
        r = subprocess.run(cmd, cwd=str(src), env=env, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout[-3000:], r.stderr[-3000:])
            print("FAIL: the converter")
            return 1
        print(f"converted: {gguf_path.stat().st_size / 1e6:.1f} MB")

        rng = random.Random(args.seed)
        toks, orders = make_prompt(rng, info["vocab"])
        inp = work / "prompt.txt"
        inp.write_text("".join(f"{t} {o}\n" for t, o in zip(toks, orders)), encoding="ascii")
        out = work / "scores.txt"
        r = subprocess.run([args.test_clef, str(gguf_path), str(inp), str(out)], capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout[-2000:], r.stderr[-3000:])
            print("FAIL: test-clef")
            return 1
        got = np.array([float(x) for x in out.read_text().split()], dtype=np.float64)

        from transformers import Qwen3_5ForCausalLM
        model = Qwen3_5ForCausalLM.from_pretrained(ck, dtype=torch.float32).eval()
        with torch.no_grad():
            ids = torch.tensor([toks])
            hidden = model.model(input_ids=ids).last_hidden_state[0]
            t = {k: v.float() for k, v in load_file(str(ck / "joint_head.safetensors")).items()}
            want = head_reference(hidden, model.lm_head.weight.float(), toks, orders, t).numpy().astype(np.float64)

        print("reference:", np.round(want, 5))
        print("engine:   ", np.round(got, 5))
        diff = float(np.abs(want - got).max())
        print(f"max |diff| = {diff:.3e} (tolerance {args.tol:g}); score spread {float(want.max() - want.min()):.3f}")
        ok = len(got) == len(want) and diff <= args.tol and float(want.max() - want.min()) > 0.05
        print("PASS" if ok else "FAIL")
        return 0 if ok else 1
    finally:
        if not args.work:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
