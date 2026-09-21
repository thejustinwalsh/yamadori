#!/usr/bin/env python
"""Rank fusion and honest confidence for code retrieval.

WHY THIS EXISTS

Measured on this stack, single-answer precision is bad and recall is fine:
the correct file was at rank 1 in 3 of 11 queries, but inside the top 5 in 7
of 11. The information is in the candidate set; the ORDER is the unreliable
part. A tool that returns one confident answer is therefore advertising a
precision it does not have, and a caller that trusts it inherits the error.

So the tool stops pretending. It returns a ranked set split by how much
independent evidence supports each hit, and lets the model -- which is far
better at disambiguation than a cosine score -- make the final call.

WHY AGREEMENT RATHER THAN A SCORE THRESHOLD

Cosine similarity barely separates good from bad on this corpus: the worst
genuine query scored 0.476 and the best nonsense query 0.445. Any threshold in
that gap is fitted to noise. Cross-encoder scores are worse -- uncalibrated,
landing near 1e-13 on correctly ordered results.

Agreement between INDEPENDENT retrievers is a better signal than any single
score, because their failure modes differ. Embeddings miss when the query uses
different vocabulary; lexical search misses when it uses the same vocabulary as
everything else; symbol lookup misses anything that is not a declaration. A
file all three surface is supported by three different kinds of evidence. This
is standard rank fusion (RRF), and it needs no GPU and no calibration.

Published work points the same way: Cursor reports ~+12.5% from combining
semantic with exact search rather than choosing between them.
"""
from __future__ import annotations

import math
import re
from collections import Counter

# Deliberately small. These are words that carry no signal in a code query and
# appear in nearly every chunk, so they only add noise to lexical scoring.
STOP = set("""the a an is are was were be been being of for to in on at by with from
as it its this that these those and or not if then else when where which who what how
we you they i return returns get gets set sets new create creates make makes
use uses used using into out up down all any some each per via than more most
function method class type interface const let var export default import
number string boolean void null undefined true false object array value values""".split())

_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")


def content_words(text: str) -> list[str]:
    """Content words, with camelCase and snake_case split into parts.

    `packGlyphAtlas` has to match a query saying "pack glyph atlas", or lexical
    search loses every multi-word identifier -- which in this corpus is most of
    them.
    """
    out = []
    for w in _WORD.findall(text):
        parts = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", w).replace("_", " ").split()
        out.extend(p.lower() for p in parts if len(p) > 2 and p.lower() not in STOP)
    return out


class Lexical:
    """BM25 over the same chunks the embeddings index.

    Built once and reused: the document frequencies do not change between
    queries, and rebuilding them per call made lexical search slower than the
    GPU it was supposed to undercut.
    """

    def __init__(self, docs: list[tuple[str, str]]):
        self.paths = [p for p, _ in docs]
        self.toks = [Counter(content_words(t)) for _, t in docs]
        self.lens = [sum(t.values()) or 1 for t in self.toks]
        self.avg = sum(self.lens) / max(len(self.lens), 1)
        self.df: Counter = Counter()
        for t in self.toks:
            for term in t:
                self.df[term] += 1
        self.n = len(docs)

    def search(self, query: str, top_k: int) -> list[int]:
        terms = content_words(query)
        if not terms:
            return []
        k1, b = 1.5, 0.75
        scored = []
        for i, t in enumerate(self.toks):
            s = 0.0
            for term in terms:
                f = t.get(term, 0)
                if not f:
                    continue
                idf = math.log(1 + (self.n - self.df[term] + 0.5) / (self.df[term] + 0.5))
                s += idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * self.lens[i] / self.avg))
            if s > 0:
                scored.append((s, i))
        scored.sort(key=lambda x: -x[0])
        return [i for _, i in scored[:top_k]]


# Standard RRF constant. Its job is to stop rank 1 from dominating so heavily
# that agreement further down carries no weight; 60 is the value from the
# original formulation and there is no reason to tune it on 11 queries.
RRF_K = 60


def fuse(rankings: dict[str, list[str]], k: int = RRF_K) -> list[dict]:
    """Reciprocal Rank Fusion over named rankings of paths.

    Each ranking is DEDUPED BY PATH first. Without that, a retriever that
    returns several chunks from one file votes for it several times, and the
    row ends up with found_by = ["semantic", "semantic"] -- two votes from one
    retriever, which then reads as agreement between two. That turns "how many
    chunks matched" into a confidence signal, which it is not.
    """
    acc: dict[str, float] = {}
    who: dict[str, set[str]] = {}
    best: dict[str, int] = {}
    for name, paths in rankings.items():
        seen: set[str] = set()
        rank = 0
        for path in paths:
            if path in seen:
                continue
            seen.add(path)
            rank += 1
            acc[path] = acc.get(path, 0.0) + 1.0 / (k + rank)
            who.setdefault(path, set()).add(name)
            best[path] = min(best.get(path, 10 ** 6), rank)

    rows = [{"path": p, "score": acc[p], "found_by": sorted(who[p]),
             "best_rank": best[p]} for p in acc]
    # Vote count first, RRF as the tiebreak. This is NOT plain RRF ordering --
    # it deliberately puts agreement ahead of any single retriever's rank.
    rows.sort(key=lambda r: (-len(r["found_by"]), -r["score"]))
    return rows


# Ranking retrievers only. `symbol` is deliberately excluded: it is a boolean
# predicate ("a declaration with this name exists here"), not a ranker, and its
# rows come back in table order. Feeding it into rank fusion made both its RRF
# contribution and its rank-based promotion depend on SQLite's insertion order.
# It is carried as a tag instead -- see tier().
RANKERS = ("semantic", "lexical")


def tier(row: dict, symbol_hit: bool = False) -> str:
    """Bucket a result by what is actually known about it.

    Each tier states a condition the caller can act on, rather than a score:

      exact        a declaration with this name exists here. High precision by
                   construction -- it is the same evidence find_definition uses.
      close        BOTH rankers surfaced it. Two different failure modes had to
                   agree, which is the real confidence signal here.
      alternative  one ranker only.

    The bar does not move with how many retrievers happened to return rows, so
    the same label means the same thing on every query.
    """
    if symbol_hit:
        return "exact"
    if len(set(row["found_by"]) & set(RANKERS)) >= len(RANKERS):
        return "close"
    return "alternative"
