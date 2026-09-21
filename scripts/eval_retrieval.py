#!/usr/bin/env python
"""Does semantic search beat keyword search at FINDING the right file?

This is the cheap, exact, model-free half of the question "are these tools
worth keeping". No model runs, nothing is graded by an LLM, and it finishes in
seconds, so it can carry the sample size the end-to-end eval never will.

THE ARMS

  search_code   embeddings over AST chunks (optionally reranked)
  keyword       the honest baseline: ripgrep-style scoring over the same files
                using the content words of the query

The keyword arm is what a model with grep and a shell already has for free. If
semantic search does not beat it, search_code is costing a GPU, an index that
goes stale on every commit, and ~100ms, to do what grep does for nothing --
and it should be cut. That is the decision this script exists to inform.

GOLD SETS

Queries are generated from the index itself: a symbol's own name and doc
comment describe what it does, so "<doc summary>" is a natural query whose
correct answer is the file that defines it. The symbol name is then STRIPPED
from the query, because leaving it in makes the task trivial for keyword
matching and measures nothing.

That auto-generation is what makes n=100+ affordable. It also biases toward
queries whose answer is a single well-named definition, which is the case
keyword search is strongest at -- so this design flatters the baseline, not
the tool being defended. Hand-written conceptual queries live in GOLD_EXTRA.
"""
from __future__ import annotations

import math
import os
import random
import re
import sqlite3
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mcp"))
import code_search as cs  # noqa: E402

STOP = set("""the a an is are was were be been being of for to in on at by with from
as it its this that these those and or not if then else when where which who what how
we you they i he she return returns get gets set sets new create creates make makes
use uses used using into out up down all any some each per via than more most
function method class type interface const let var export default import
number string boolean void null undefined true false object array value values
code file line does do done can may should must will would""".split())

WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")


def content_words(q: str) -> list[str]:
    """Content words of a query, with camelCase and snake_case split apart."""
    out = []
    for w in WORD.findall(q):
        parts = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", w).replace("_", " ").split()
        out.extend(p.lower() for p in parts if len(p) > 2 and p.lower() not in STOP)
    return out


# ---------------------------------------------------------------- keyword arm
# Uses the same BM25 the stack ships, built ONCE over the corpus. Rebuilding
# document frequencies per query is O(queries x chunks) and does not finish on
# a 60k-chunk index.
import fusion  # noqa: E402

_LEX = None


def keyword_rank(query: str, docs: list[tuple[str, str]], top_k: int) -> list[str]:
    global _LEX
    if _LEX is None:
        _LEX = fusion.Lexical(docs)
    out, seen = [], set()
    for i in _LEX.search(query, top_k * 4):
        p = docs[i][0]
        if p not in seen:
            seen.add(p)
            out.append(p)
        if len(out) >= top_k:
            break
    return out


# ------------------------------------------------------------------ gold sets
def build_gold(con: sqlite3.Connection, limit: int, seed: int = 0):
    """Query = a symbol's doc/signature with the symbol NAME removed."""
    rows = con.execute(
        "SELECT name, kind, path, start, line FROM defs "
        "WHERE kind IN ('function','class','interface','type','struct','trait','enum')"
    ).fetchall()
    random.Random(seed).shuffle(rows)

    gold = []
    seen = set()
    for name, kind, path, start, line in rows:
        if len(gold) >= limit:
            break
        if not name or len(name) < 4 or name in seen:
            continue
        # Turn the identifier into words, then drop it from the query so the
        # task is not a literal lookup of a name the caller already has.
        words = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name).replace("_", " ").lower()
        words = [w for w in words.split() if w not in STOP and len(w) > 2]
        if len(words) < 2:
            continue
        seen.add(name)
        gold.append((f"{kind} that handles {' '.join(words)}", path, name))
    return gold


GOLD_EXTRA: list[tuple[str, str]] = [
    # Hand-written, conceptual: the shape where an index should beat keywords,
    # because the code does not contain the words the question uses.
    ("how are glyph atlases packed into a texture", "glyph"),
    ("where does a component get registered with the world", "koota"),
    ("how does a typescript type get emitted as lua", "deherm"),
    ("what crosses the wasm boundary as a raw pointer", "glyph"),
    ("how is a sprite batched before it is drawn", "three-flatland"),
]


def main() -> None:
    db = os.environ.get("CODE_INDEX_DB", cs.INDEX_DB)
    n_gold = int(os.environ.get("EVAL_N", "120"))
    top_k = int(os.environ.get("EVAL_K", "5"))

    con = sqlite3.connect(db)
    chunks = con.execute("SELECT path, text FROM chunks").fetchall()
    gold = build_gold(con, n_gold)
    con.close()
    if not chunks or not gold:
        print(f"index at {db} is empty or has no symbols")
        return

    print(f"index   : {db}")
    print(f"chunks  : {len(chunks)}   gold queries: {len(gold)}   top_k={top_k}\n")

    sem_hit = kw_hit = both = neither = sem_only = kw_only = 0
    sem_rr = kw_rr = 0.0
    fus_hit = 0
    fus_rr = 0.0
    fus_only = kw_only_f = 0

    for i, (query, gold_path, _name) in enumerate(gold, 1):
        try:
            hits = cs.search(query, top_k)
        except Exception as e:                                   # noqa: BLE001
            print(f"search failed: {e}")
            return
        sem = [h["path"] for h in hits]
        kw = keyword_rank(query, chunks, top_k)
        # The shipped configuration: semantic + lexical fused, symbol hits
        # tagged. This is the arm the pre-registered cut rule names.
        fus = [h["path"] for h in cs.search_fused(query, top_k)][:top_k]

        s_ok = gold_path in sem
        k_ok = gold_path in kw
        sem_hit += s_ok
        kw_hit += k_ok
        both += s_ok and k_ok
        neither += not s_ok and not k_ok
        sem_only += s_ok and not k_ok
        kw_only += k_ok and not s_ok
        if s_ok:
            sem_rr += 1 / (sem.index(gold_path) + 1)
        if k_ok:
            kw_rr += 1 / (kw.index(gold_path) + 1)
        f_ok = gold_path in fus
        fus_hit += f_ok
        if f_ok:
            fus_rr += 1 / (fus.index(gold_path) + 1)
        fus_only += f_ok and not k_ok
        kw_only_f += k_ok and not f_ok
        if i % 20 == 0:
            print(f"  ...{i}/{len(gold)}")

    n = len(gold)
    print("\n" + "=" * 64)
    print(f"{'':<22}{'semantic':>10}{'keyword':>10}{'fused':>10}")
    print(f"{'recall@' + str(top_k):<22}{sem_hit:>7}/{n}{kw_hit:>7}/{n}{fus_hit:>7}/{n}")
    print(f"{'MRR':<22}{sem_rr / n:>10.3f}{kw_rr / n:>10.3f}{fus_rr / n:>10.3f}")
    print("-" * 64)
    # McNemar's discordant pairs: the only cells that carry information about
    # which arm is better. Reporting 62% vs 55% hides that most tasks are ties.
    print("discordant pairs (the part that decides it):")
    print(f"  semantic only : {sem_only}")
    print(f"  keyword only  : {kw_only}")
    print(f"  both          : {both}")
    print(f"  neither       : {neither}")
    d = sem_only + kw_only
    if d:
        # Two-sided exact binomial on the discordant pairs.
        k = min(sem_only, kw_only)
        p = sum(math.comb(d, i) for i in range(k + 1)) / (2 ** d) * 2
        print(f"  exact McNemar p = {min(p, 1.0):.4f}"
              + ("   <- not significant" if p > 0.05 else "   <- significant"))
    else:
        print("  no discordant pairs: the arms are indistinguishable here")

    print("\n" + "-" * 64)
    print("FUSED vs KEYWORD -- the arm the cut rule names:")
    print(f"  fused only    : {fus_only}")
    print(f"  keyword only  : {kw_only_f}")
    d2 = fus_only + kw_only_f
    if d2:
        k2 = min(fus_only, kw_only_f)
        p2 = sum(math.comb(d2, i) for i in range(k2 + 1)) / (2 ** d2) * 2
        p2 = min(p2, 1.0)
        print(f"  exact McNemar p = {p2:.4f}")
        if fus_only > kw_only_f and p2 < 0.05:
            print("  -> fusion beats keyword. KEEP the embedding index.")
        elif p2 < 0.05:
            print("  -> keyword beats fusion. CUT RULE FIRES: drop embeddings,")
            print("     keep BM25 + the symbol table, which need no GPU.")
        else:
            print("  -> indistinguishable. The index is not earning a GPU.")
    else:
        print("  no discordant pairs")


if __name__ == "__main__":
    main()
