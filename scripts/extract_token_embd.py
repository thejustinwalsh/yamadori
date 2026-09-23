#!/usr/bin/env python
"""Extract the model's OWN input embeddings, once, offline.

    python scripts/extract_token_embd.py [--gguf PATH] [--out index/token_embd.npz]

WHAT THIS IS FOR

Two consumers need the 27B's embedding space itself -- not a list, and not the
retrieval embedder's space:

  concept_seed   A random direction in the model's space, snapped to the
                 nearest whole-word token in the model's vocabulary. The
                 candidate set is every whole-word token the model has
                 (tens of thousands), not a curated list.
  bonsai viz     A FROZEN 3-component PCA basis of `token_embd.weight`, so each
                 generated token id is a fixed point in the model's own space
                 (design/BONSAI-VIZ.md, "The 3D basis, and why it must be
                 frozen"). Fitted once here, never refitted.

THE FORMAT

`token_embd.weight` is 5120 x 248,320, ggml type 143 = PTQ1_0, which is
Prism-private: 128 ternary weights per 28-byte block -- 24 bytes of base-3
trits (5 per byte), 2 bytes of trits (4 per byte), one fp16 scale. The decode
below is a numpy transcription of `dequantize_row_ptq1_0` in the PrismML fork
(ggml/src/ggml-quants.c), stage table {32, 16, 8}.

The matrix is stored Hadamard-ROTATED: the header lists it under
`prism.hadamard.inverse_weight_names`, with a normalized Sylvester-Walsh
transform and explicit signs. That rotation is orthonormal, so inner products,
norms, nearest neighbours and PCA variance are all unchanged by it. Nothing
here undoes it, and nothing here needs to. The PCA basis is in rotated
coordinates, which is fine because every point projected through it is too.

THE CHECK THAT GATES THE WRITE

A wrong decode does not crash; it produces a matrix of noise whose nearest
neighbours are random. So before writing, known words must find related words
among their nearest neighbours. If they do not, this exits non-zero and writes
nothing -- the same rule as PROTOCOL rule 1 (an index of zero vectors once
reported success).
"""
from __future__ import annotations

import argparse
import os
import re
import struct
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
DEFAULT_GGUF = ("C:/Users/jwals/textgen/user_data/models/"
                "Ternary-Bonsai-2-27B-Abliterated-PTQ1_0.gguf")
DEFAULT_OUT = os.path.join(ROOT, "index", "token_embd.npz")

PTQ1_0 = 143
BLOCK_WEIGHTS = 128
BLOCK_BYTES = 28          # qs[24] + qh[2] + fp16 d
POW3 = np.array([1, 3, 9, 27, 81, 243], dtype=np.uint8)

# A whole-word token: GPT-2 byte-level BPE marks a leading space with U+0120.
# Lowercase letters only, so the candidates are words rather than fragments,
# identifiers or numbers.
WORD = re.compile("^\u0120([a-z]{4,14})$")


# ---------------------------------------------------------------------------
# GGUF header. Only what is needed: metadata values and tensor offsets.
# ---------------------------------------------------------------------------
_SCALAR = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?",
           10: "Q", 11: "q", 12: "d"}


def read_header(path: str) -> tuple[dict, dict, int]:
    """(metadata, tensors {name: (dims, type, offset)}, data_start)."""
    f = open(path, "rb")

    def rd(fmt):
        return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))[0]

    def rstr():
        return f.read(rd("Q")).decode("utf-8", "replace")

    def rval(t):
        if t == 8:
            return rstr()
        if t == 9:
            et, n = rd("I"), rd("Q")
            return [rval(et) for _ in range(n)]
        return rd(_SCALAR[t])

    if f.read(4) != b"GGUF":
        raise SystemExit(f"{path} is not a GGUF file")
    version, n_tensors, n_kv = rd("I"), rd("Q"), rd("Q")
    if version != 3:
        raise SystemExit(f"GGUF version {version}; this reader knows 3")
    meta = {}
    for _ in range(n_kv):
        k = rstr()
        meta[k] = rval(rd("I"))
    tensors = {}
    for _ in range(n_tensors):
        name = rstr()
        dims = [rd("Q") for _ in range(rd("I"))]
        tensors[name] = (dims, rd("I"), rd("Q"))
    align = int(meta.get("general.alignment", 32))
    pos = f.tell()
    f.close()
    return meta, tensors, (pos + align - 1) // align * align


# ---------------------------------------------------------------------------
# PTQ1_0 decode. Matches dequantize_row_ptq1_0 value for value.
# ---------------------------------------------------------------------------
def _trits(bytes_: np.ndarray, count: int) -> np.ndarray:
    """(B, c) packed bytes -> (B, count*c) trits in {-1,0,1}, in C order:
    for n in range(count): for m in range(c)."""
    q = bytes_[:, None, :] * POW3[:count, None]          # uint8, wraps like C
    return ((q.astype(np.uint16) * 3) >> 8).astype(np.int8).reshape(
        bytes_.shape[0], -1) - 1


def decode_blocks(raw: np.ndarray) -> np.ndarray:
    """(B, 28) uint8 blocks -> (B, 128) float32."""
    qs, qh = raw[:, :24], raw[:, 24:26]
    d = raw[:, 26:28].copy().view(np.float16).astype(np.float32)
    # Stage 32 never fits in 24 bytes; stage 16 takes bytes 0-15, stage 8
    # takes 16-23, then qh. Same loop bounds as the C.
    vals = np.concatenate([_trits(qs[:, 0:16], 5), _trits(qs[:, 16:24], 5),
                           _trits(qh, 4)], axis=1)
    return vals.astype(np.float32) * d


def rows(path: str, offset: int, n_rows: int, n_cols: int,
         start: int, stop: int) -> np.ndarray:
    """Decode rows [start, stop) of a PTQ1_0 matrix, memory-mapped."""
    per_row = n_cols // BLOCK_WEIGHTS * BLOCK_BYTES
    mm = np.memmap(path, dtype=np.uint8, mode="r", offset=offset,
                   shape=(n_rows * per_row,))
    raw = np.asarray(mm[start * per_row:stop * per_row]).reshape(-1, BLOCK_BYTES)
    return decode_blocks(raw).reshape(stop - start, n_cols)


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--gguf", default=DEFAULT_GGUF)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--chunk", type=int, default=4096)
    a = ap.parse_args()

    t0 = time.time()
    meta, tensors, data_start = read_header(a.gguf)
    if "token_embd.weight" not in tensors:
        raise SystemExit("no token_embd.weight tensor in this GGUF")
    (n_cols, n_rows), ttype, off = tensors["token_embd.weight"]
    if ttype != PTQ1_0:
        raise SystemExit(f"token_embd.weight is ggml type {ttype}; this "
                         f"decoder handles only PTQ1_0 ({PTQ1_0})")
    tokens = meta["tokenizer.ggml.tokens"]
    if len(tokens) != n_rows:
        raise SystemExit(f"{len(tokens)} tokens but {n_rows} embedding rows")
    offset = data_start + off
    print(f"  {os.path.basename(a.gguf)}: token_embd {n_rows:,} x {n_cols}, "
          f"PTQ1_0, hadamard-rotated="
          f"{'token_embd.weight' in meta.get('prism.hadamard.inverse_weight_names', [])}")

    word_ids = [i for i, t in enumerate(tokens) if WORD.match(t)]
    words = [WORD.match(tokens[i]).group(1) for i in word_ids]
    print(f"  {len(word_ids):,} whole-word tokens of {n_rows:,}")

    # One pass: accumulate the covariance for the PCA over EVERY token, and
    # keep the normalised word rows for concept_seed.
    mean = np.zeros(n_cols, np.float64)
    gram = np.zeros((n_cols, n_cols), np.float64)
    word_set = {i: k for k, i in enumerate(word_ids)}
    word_mat = np.zeros((len(word_ids), n_cols), np.float16)
    norms_all = np.zeros(n_rows, np.float32)
    for s in range(0, n_rows, a.chunk):
        e = min(n_rows, s + a.chunk)
        x = rows(a.gguf, offset, n_rows, n_cols, s, e)
        norms_all[s:e] = np.linalg.norm(x, axis=1)
        mean += x.sum(axis=0, dtype=np.float64)
        gram += x.T.astype(np.float64) @ x
        for r in range(s, e):
            k = word_set.get(r)
            if k is not None:
                v = x[r - s]
                word_mat[k] = (v / (np.linalg.norm(v) or 1.0)).astype(np.float16)
        print(f"\r  decoded {e:,}/{n_rows:,}  {time.time() - t0:.0f}s",
              end="", flush=True)
    print()

    zero = int((norms_all == 0).sum())
    print(f"  row norms: min {norms_all.min():.4f}  median "
          f"{np.median(norms_all):.4f}  max {norms_all.max():.4f}  "
          f"zero rows {zero:,}")

    mean /= n_rows
    cov = gram / n_rows - np.outer(mean, mean)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1][:3]
    basis = evecs[:, order].T.astype(np.float32)          # (3, n_cols)
    explained = (evals[order] / evals.sum()).astype(np.float32)
    print(f"  PCA top-3 explained variance: {np.round(explained, 4).tolist()}")

    # THE GATE: known words must land near related words. A wrong decode is
    # noise, and noise has random neighbours.
    probes = {"king": {"queen", "kings", "prince", "royal", "monarch",
                       "kingdom", "throne", "emperor", "lord"},
              "water": {"waters", "liquid", "rain", "river", "ocean",
                        "fluid", "moisture", "lake", "aqueous", "wet"},
              "three": {"four", "five", "two", "seven", "six", "eight",
                        "nine", "thirty", "third"},
              "green": {"yellow", "blue", "purple", "orange", "grey",
                        "brown", "greens", "gray", "pink", "white"}}
    wf = word_mat.astype(np.float32)
    index = {w: k for k, w in enumerate(words)}
    hits = 0
    for w, related in probes.items():
        if w not in index:
            print(f"  probe {w!r} is not a whole-word token; skipped")
            continue
        sims = wf @ wf[index[w]]
        top = [words[j] for j in np.argsort(-sims)[1:16]]
        ok = bool(related & set(top))
        hits += ok
        print(f"  {'ok  ' if ok else 'MISS'} {w:>6}: {', '.join(top[:10])}")
    if hits < 3:
        print(f"\n  REFUSING TO WRITE: only {hits}/4 probes found a related "
              "word among 15 neighbours. The decode is wrong, or this is not "
              "the model it claims to be.")
        return 1

    # Every token's frozen 3D coordinate, for the visualisation.
    coords = np.zeros((n_rows, 3), np.float16)
    for s in range(0, n_rows, a.chunk):
        e = min(n_rows, s + a.chunk)
        x = rows(a.gguf, offset, n_rows, n_cols, s, e)
        coords[s:e] = ((x - mean.astype(np.float32)) @ basis.T).astype(np.float16)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez(a.out, words=np.array(words), word_ids=np.array(word_ids, np.int32),
             word_mat=word_mat, pca_mean=mean.astype(np.float32),
             pca_basis=basis, pca_explained=explained, token_xyz=coords,
             source=os.path.basename(a.gguf), n_vocab=n_rows, dim=n_cols,
             probe_hits=hits)
    print(f"  wrote {a.out}  ({os.path.getsize(a.out) / 1e6:.0f} MB) in "
          f"{time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
