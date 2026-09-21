#!/usr/bin/env python
"""Does the cross-encoder actually improve the ordering, or just cost time?

search_code runs embeddings for recall and a reranker for precision. The
reranker is the expensive half and its scores are uncalibrated, so the only
thing that can justify it is whether it moves the RIGHT file up the list.

Each case is a query plus a path fragment that a correct answer must contain.
We record where that file lands under embedding order and under rerank order.
Lower rank is better; 0 means it was not in the candidate set at all, which no
amount of reranking can fix -- that is a recall failure, not a precision one.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mcp"))
import code_search as cs  # noqa: E402

CASES = [
    ("how are dashed line uniforms refreshed",        "WebGLMaterials"),
    ("where is the shader program cache keyed",       "WebGLPrograms"),
    ("morph target attribute binding",                "WebGLMorphtargets"),
    ("how does it decide to recompile a program",     "WebGLRenderer"),
    ("how are bone matrices uploaded for skinning",   "Skeleton"),
    ("where are draw calls actually issued",          "WebGLBufferRenderer"),
    ("how is the depth texture allocated",            "WebGLTextures"),
    ("clipping planes uniform setup",                 "WebGLClipping"),
    ("how does raycasting test a mesh",               "Mesh"),
    ("frustum culling of objects before render",      "Frustum"),
    ("how are instanced attributes uploaded",         "WebGLAttributes"),
    ("spherical harmonics for light probes",          "SphericalHarmonics3"),
]


def rank_of(order: list[str], frag: str) -> int:
    """1-based rank of the first path containing frag, else 0."""
    for i, p in enumerate(order, 1):
        if frag.lower() in p.lower():
            return i
    return 0


def main() -> None:
    chunks, mat = cs.load_index()
    if mat is None:
        print("index is empty")
        return

    rows = []
    for query, frag in CASES:
        qv = cs.embed([query], is_query=True)[0]
        sims = mat @ qv
        n = min(cs.CANDIDATES, len(chunks))
        top = np.argpartition(-sims, n - 1)[:n]
        top = top[np.argsort(-sims[top])]
        cand = [chunks[i] for i in top]

        embed_order = [c.path for c in cand]
        docs = [f"{c.path}:{c.start}-{c.end}\n{c.text[:cs.RERANK_DOC_CHARS]}"
                for c in cand]

        t0 = time.time()
        try:
            ranked = cs.rerank(query, docs, len(docs))
            rr_order = [cand[i].path for i, _ in ranked]
        except Exception as e:                                   # noqa: BLE001
            print(f"rerank failed: {e}")
            return
        ms = (time.time() - t0) * 1000

        re, rr = rank_of(embed_order, frag), rank_of(rr_order, frag)
        rows.append((query, frag, re, rr, ms))

    print(f"{'embed':>5} {'rerank':>7} {'delta':>6} {'ms':>6}  query")
    print("-" * 78)
    better = worse = same = 0
    for query, frag, re, rr, ms in rows:
        # Only count cases the embedding stage actually retrieved; a miss there
        # is a recall problem and says nothing about the reranker.
        if re and rr:
            d = re - rr
            better += d > 0
            worse += d < 0
            same += d == 0
            ds = f"{d:+d}"
        else:
            ds = "n/a"
        print(f"{re or '-':>5} {rr or '-':>7} {ds:>6} {ms:>6.0f}  {query[:44]}")

    scored = [(re, rr) for _, _, re, rr, _ in rows if re and rr]
    print("-" * 78)
    print(f"comparable cases : {len(scored)} of {len(rows)}")
    print(f"rerank better    : {better}")
    print(f"rerank worse     : {worse}")
    print(f"unchanged        : {same}")
    if scored:
        print(f"mean rank embed  : {sum(a for a, _ in scored)/len(scored):.2f}")
        print(f"mean rank rerank : {sum(b for _, b in scored)/len(scored):.2f}")
        t1e = sum(1 for a, _ in scored if a == 1)
        t1r = sum(1 for _, b in scored if b == 1)
        print(f"correct at #1    : embed {t1e}/{len(scored)}   rerank {t1r}/{len(scored)}")
        t3e = sum(1 for a, _ in scored if a <= 3)
        t3r = sum(1 for _, b in scored if b <= 3)
        print(f"correct in top-3 : embed {t3e}/{len(scored)}   rerank {t3r}/{len(scored)}")
        # DEFAULT_TOP_K is what search_code actually returns, so this is the
        # only cutoff that decides whether the reranker earns its latency.
        k = cs.DEFAULT_TOP_K
        tke = sum(1 for a, _ in scored if a <= k)
        tkr = sum(1 for _, b in scored if b <= k)
        print(f"correct in top-{k} : embed {tke}/{len(scored)}   rerank {tkr}/{len(scored)}"
              f"   <-- the shipped cutoff")
    print(f"mean rerank cost : {sum(r[4] for r in rows)/len(rows):.0f} ms")


if __name__ == "__main__":
    main()
