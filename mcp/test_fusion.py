#!/usr/bin/env python
"""Rank fusion and result tiers, asserted directly. No index, no GPU.

WHAT THIS IS GATING

`fusion.py` turns several retrievers' rankings into one list whose labels
mean the same thing on every query. Its promises:

  1. `content_words` splits camelCase and snake_case, so `packGlyphAtlas`
     matches "pack glyph atlas", and drops stop words.
  2. BM25 ranks a document that uses a rare query term above one that only
     shares common ones, and returns nothing for a query of stop words.
  3. `fuse` dedupes each ranking BY PATH before voting, so three chunks from
     one file are one vote -- never agreement between retrievers.
  4. Agreement outranks any single retriever's rank.
  5. Tiers: a symbol hit is `exact`; both rankers is `close`; one is
     `alternative`. `symbol` is not a ranker and cannot make a row `close`.

WHAT IS COVERED ELSEWHERE

`mcp/test_tools.py` exercises this module end to end through
`code_search.search_code` on a fixture index, and breaks `content_words` to
simulate a lexical outage. This file tests the arithmetic underneath.
"""
from __future__ import annotations

import json
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import fusion  # noqa: E402

_results: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> bool:
    _results.append((bool(ok), name, detail))
    return bool(ok)


# ---------------------------------------------------------------------------


def test_content_words_splits_identifiers():
    check(fusion.content_words("packGlyphAtlas") == ["pack", "glyph", "atlas"],
          "camelCase is split", str(fusion.content_words("packGlyphAtlas")))
    check(fusion.content_words("load_font_face") == ["load", "font", "face"],
          "snake_case is split", str(fusion.content_words("load_font_face")))
    check(fusion.content_words("the function returns a value") == [],
          "stop words and code keywords are dropped",
          str(fusion.content_words("the function returns a value")))
    check(fusion.content_words("go to db x") == [],
          "words of two letters or fewer are dropped")
    check(fusion.content_words("Vec3 parse2d") == ["vec3", "parse2d"],
          "digits stay inside a word", str(fusion.content_words("Vec3 parse2d")))
    check(fusion.content_words("") == [] and fusion.content_words("!!!") == [],
          "empty or punctuation-only text has no words")


def test_bm25_prefers_the_rare_matching_term():
    docs = [("a.ts", "render mesh render mesh render"),
            ("b.ts", "packGlyphAtlas builds the glyph atlas"),
            ("c.ts", "render loop"),
            ("d.ts", "unrelated text about sockets")]
    lex = fusion.Lexical(docs)
    hits = [docs[i][0] for i in lex.search("glyph atlas", top_k=5)]
    check(hits == ["b.ts"], "only the document with the terms is returned", str(hits))
    hits = [docs[i][0] for i in lex.search("render glyph", top_k=5)]
    check(hits and hits[0] == "b.ts",
          "a rare term outweighs a common one repeated", str(hits))
    check(set(hits) == {"a.ts", "b.ts", "c.ts"}, "every matching document is found",
          str(hits))
    check(lex.search("the function", top_k=5) == [],
          "a query of stop words returns nothing rather than everything")
    check(len(lex.search("render", top_k=1)) == 1, "top_k is respected")


def test_bm25_survives_degenerate_corpora():
    try:
        check(fusion.Lexical([]).search("anything", top_k=3) == [],
              "an empty corpus returns no hits")
        lex = fusion.Lexical([("e.ts", ""), ("f.ts", "the a an")])
        check(lex.search("anything", top_k=3) == [],
              "documents with no content words return no hits")
    except Exception as e:                                       # noqa: BLE001
        check(False, "degenerate corpora do not raise", f"{type(e).__name__}: {e}")


def test_fuse_counts_one_vote_per_retriever_per_file():
    rows = fusion.fuse({"semantic": ["a.ts", "a.ts", "a.ts", "b.ts"],
                        "lexical": ["b.ts"]})
    by = {r["path"]: r for r in rows}
    check(by["a.ts"]["found_by"] == ["semantic"],
          "three chunks of one file are one vote, not three",
          json.dumps(by["a.ts"]))
    check(by["b.ts"]["best_rank"] == 1 and by["b.ts"]["found_by"] == ["lexical",
                                                                        "semantic"],
          "a duplicate does not push later files down the ranking",
          json.dumps(by["b.ts"]))
    expected = 1 / (60 + 2) + 1 / (60 + 1)
    check(abs(by["b.ts"]["score"] - expected) < 1e-12,
          "the RRF score is 1/(k+rank) summed, with deduped ranks",
          f"{by['b.ts']['score']} vs {expected}")


def test_agreement_outranks_a_single_high_rank():
    rows = fusion.fuse({"semantic": ["solo.ts", "x.ts", "y.ts", "both.ts"],
                        "lexical": ["z.ts", "w.ts", "v.ts", "both.ts"]})
    check(rows[0]["path"] == "both.ts",
          "a file two retrievers agree on leads, even at rank 4 in each",
          str([r["path"] for r in rows[:3]]))
    single = [r for r in rows if len(r["found_by"]) == 1]
    scores = [r["score"] for r in single]
    check(scores == sorted(scores, reverse=True),
          "within one vote count, RRF score orders the rest")
    check(fusion.fuse({}) == [] and fusion.fuse({"semantic": []}) == [],
          "no rankings fuse to nothing")


def test_tiers_mean_the_same_thing_on_every_query():
    both = {"found_by": ["lexical", "semantic"]}
    one = {"found_by": ["semantic"]}
    check(fusion.tier(one, symbol_hit=True) == "exact", "a symbol hit is exact")
    check(fusion.tier(both) == "close", "both rankers is close")
    check(fusion.tier(one) == "alternative", "one ranker is alternative")
    check(fusion.tier({"found_by": ["semantic", "symbol"]}) == "alternative",
          "`symbol` is not a ranker and cannot make a row close")
    check(fusion.tier({"found_by": []}) == "alternative",
          "no ranker at all is still only alternative")


def main() -> int:
    for fn in (test_content_words_splits_identifiers,
               test_bm25_prefers_the_rare_matching_term,
               test_bm25_survives_degenerate_corpora,
               test_fuse_counts_one_vote_per_retriever_per_file,
               test_agreement_outranks_a_single_high_rank,
               test_tiers_mean_the_same_thing_on_every_query):
        print(f"\n--- {fn.__name__} ---")
        n0 = len(_results)
        try:
            fn()
        except Exception:                                        # noqa: BLE001
            check(False, f"{fn.__name__} itself raised",
                  traceback.format_exc().strip().split("\n")[-1])
        for ok, name, detail in _results[n0:]:
            print(("  pass  " if ok else "  FAIL  ") + name
                  + (f"   <- {detail}" if not ok and detail else ""))

    passed = sum(1 for ok, _, _ in _results if ok)
    total = len(_results)
    print(f"\n{'=' * 70}\n  {passed}/{total} checks passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
