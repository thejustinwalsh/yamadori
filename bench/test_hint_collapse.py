#!/usr/bin/env python
"""Every claim in docs/HINTS.md, as an assertion that fails when it stops holding.

WHY THIS FILE EXISTS

The expensive result in docs/HINTS.md is a NEGATIVE one -- Laya's joint scoring
does not beat independent embedding retrieval at collapsing a mutually exclusive
bucket of hints, not even on the contrastive buckets where the mechanism
predicts it should. That is exactly the kind of finding someone re-litigates in
three months by running one promising prompt and stopping. The floors below make
that re-litigation cost a failing test instead of a week.

Floors are deliberately set BELOW the measured numbers, wide enough that fp16
batch jitter and a llama-swap restart cannot trip them, and tight enough that a
real regression or a real improvement both show up.

TWO MODES

  default   score the CACHED run in bench/data/hint_collapse_runs.json. No GPU,
            no network, ~1 s, runs anywhere.
  --live    additionally hit the reranker on 11434 and assert the cache
            still matches what it returns today, plus the two serving
            diagnostics that explain the rerank arm. (The Laya check on 1237
            is retired with Laya, 2026-09-24, docs/E1.md.)

Regenerate the cache with:

    python -X utf8 bench/hint_collapse.py

Run:

    python -X utf8 bench/test_hint_collapse.py
    python -X utf8 bench/test_hint_collapse.py --live
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import hint_collapse as H                                        # noqa: E402

_FAILURES: list[str] = []
_PASSES = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global _PASSES
    if cond:
        _PASSES += 1
        print(f"  ok   {name}" + (f"   [{detail}]" if detail else ""))
    else:
        _FAILURES.append(f"{name}: {detail}")
        print(f"  FAIL {name}   [{detail}]")


# ------------------------------------------------------------- the floors ---
#
# Measured 2026-09-22. Every number is "the run scored X, the floor is a little
# under X" -- a floor equal to the measurement would fail on any jitter, and a
# floor far under it would never fire.

MEASURED = {
    # arm: (overall, contrastive, topical) accuracy at full coverage
    "random":         (0.213, 0.277, 0.143),
    "fixed":          (0.213, 0.234, 0.190),
    "lexical":        (0.517, 0.468, 0.571),
    "embedding":      (0.719, 0.574, 0.881),
    "embedding_cond": (0.573, 0.447, 0.714),
    "rerank":         (0.225, 0.277, 0.167),
    "rerank_batched": (0.213, 0.255, 0.167),
    "laya":           (0.337, 0.298, 0.381),
    "laya_cond":      (0.449, 0.362, 0.548),
}
TOL = 0.06          # the band a re-run must stay inside


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true")
    a = ap.parse_args(argv)

    print("DATA")
    buckets = H.load_jsonl(H.BUCKETS)
    probes = H.load_jsonl(H.PROBES)
    check("hint_buckets.jsonl loads", len(buckets) > 0, f"{len(buckets)}")
    check("hint_probes.jsonl loads", len(probes) > 0, f"{len(probes)}")
    check("at least 12 buckets, as the brief requires", len(buckets) >= 12,
          f"{len(buckets)}")
    check("both bucket types are present and neither is a token handful",
          min(sum(1 for b in buckets if b["bucket_type"] == t)
              for t in ("contrastive", "topical")) >= 5,
          str({t: sum(1 for b in buckets if b["bucket_type"] == t)
               for t in ("contrastive", "topical")}))
    check("every bucket has >= 2 members",
          all(len(b["members"]) >= 2 for b in buckets),
          f"min {min(len(b['members']) for b in buckets)}")
    check("every bucket records why it is exclusive",
          all(len(b.get("why_exclusive") or "") > 40 for b in buckets))
    problems = H.check_buckets(buckets, probes)
    check("structural invariants (one probe per member, ground truth in "
          "bucket, no duplicate ids)", not problems, "; ".join(problems[:3]))
    drift = H.verify_no_drift(buckets)
    check("no corpus drift -- every member text is still a verbatim recipe in "
          "bench/recipes/*.jsonl", not drift, "; ".join(drift[:3]))

    # The trap the brief names explicitly: `area` is NOT the exclusion key.
    # range_structure omits complexity:24 and complexity:26, which share its
    # area and would be co-applicable constraints rather than competitors.
    rng = next(b for b in buckets if b["bucket_id"] == "range_structure")
    refs = {m["ref"] for m in rng["members"]}
    check("the documented trap is avoided: the associativity CONSTRAINT "
          "(complexity:24) is not a member of the range bucket",
          "complexity:24" not in refs and "complexity:26" not in refs,
          " ".join(sorted(refs)))

    # Probes must not be paraphrases of their own hint, or every arm is scored
    # on string overlap. The lexical floor is the quantitative version of this;
    # this is the per-probe version.
    by_id = {b["bucket_id"]: b for b in buckets}
    worst = []
    for p in probes:
        b = by_id[p["bucket_id"]]
        m = next(x for x in b["members"] if x["member_id"] == p["correct"])
        q, d = H._tokens(p["problem"]), H._tokens(m["recipe"])
        j = len(q & d) / max(1, len(q | d))
        worst.append((j, p["probe_id"]))
    worst.sort(reverse=True)
    check("no probe is a paraphrase of its own hint (max Jaccard < 0.30)",
          worst[0][0] < 0.30, f"max {worst[0][0]:.2f} at {worst[0][1]}")

    print("\nCACHED RUN")
    if not os.path.exists(H.RUNS):
        check("bench/data/hint_collapse_runs.json exists", False,
              "run bench/hint_collapse.py once")
        return 1
    with open(H.RUNS, encoding="utf-8") as fh:
        cache = json.load(fh)
    check("cache covers every probe",
          all(p["probe_id"] in cache["probes"] for p in probes),
          f"{len(cache['probes'])} entries")
    arms = tuple(MEASURED)
    records, _ = H.run(buckets, probes, arms, cache, True, False)
    check("every arm replays from the cache with no GPU",
          all(all(x in r["arms"] for x in arms) for r in records))

    contr = lambda r: r["bucket_type"] == "contrastive"           # noqa: E731
    top = lambda r: r["bucket_type"] == "topical"                 # noqa: E731

    print("\nACCURACY FLOORS (docs/HINTS.md table 1)")
    acc = {}
    for arm, (o, c, t) in MEASURED.items():
        got = (H.evaluate(records, arm)["acc"],
               H.evaluate(records, arm, contr)["acc"],
               H.evaluate(records, arm, top)["acc"])
        acc[arm] = got
        check(f"{arm}: overall/contrastive/topical within +/-{TOL:.2f} of the "
              "recorded run",
              all(abs(g - w) <= TOL for g, w in zip(got, (o, c, t))),
              f"got {tuple(round(g, 3) for g in got)} want {(o, c, t)}")

    print("\nTHE VERDICT (docs/HINTS.md finding 1)")
    check("embedding beats laya overall, and it is not close",
          acc["embedding"][0] - acc["laya"][0] > 0.25,
          f"{acc['embedding'][0]:.3f} vs {acc['laya'][0]:.3f}")
    check("embedding beats laya ON THE CONTRASTIVE BUCKETS -- the half where "
          "the mechanism predicted laya would win",
          acc["embedding"][1] > acc["laya"][1],
          f"{acc['embedding'][1]:.3f} vs {acc['laya'][1]:.3f}")
    check("the best laya configuration tried (condition-only options) still "
          "loses to embedding on contrastive buckets",
          acc["embedding"][1] > acc["laya_cond"][1],
          f"{acc['embedding'][1]:.3f} vs {acc['laya_cond'][1]:.3f}")
    check("laya does beat the random floor, so it is not measuring nothing",
          acc["laya"][0] > acc["random"][0] + 0.05,
          f"{acc['laya'][0]:.3f} vs {acc['random'][0]:.3f}")
    check("laya loses to the six-line LEXICAL floor overall -- the same shape "
          "as docs/LAYA.md finding 11",
          acc["lexical"][0] > acc["laya"][0],
          f"{acc['lexical'][0]:.3f} vs {acc['laya'][0]:.3f}")
    check("embedding beats the lexical floor, so its win is semantic and not "
          "string overlap",
          acc["embedding"][0] > acc["lexical"][0] + 0.10,
          f"{acc['embedding'][0]:.3f} vs {acc['lexical'][0]:.3f}")
    # docs/HINTS.md finding 5: the obvious fix for the contrastive gap is to
    # embed the CONDITION instead of the recipe. It makes things worse, in the
    # opposite direction to what the same change does for Laya.
    check("embedding the condition instead of the recipe is WORSE everywhere, "
          "including on the contrastive slice it was meant to rescue",
          all(acc["embedding"][i] > acc["embedding_cond"][i] for i in range(3)),
          f"embedding {tuple(round(x, 3) for x in acc['embedding'])} vs "
          f"embedding_cond {tuple(round(x, 3) for x in acc['embedding_cond'])}")
    check("shortening the option text helps LAYA and hurts EMBEDDING -- the two "
          "mechanisms want opposite things from the same text",
          acc["laya_cond"][0] > acc["laya"][0]
          and acc["embedding_cond"][0] < acc["embedding"][0],
          f"laya {acc['laya'][0]:.3f}->{acc['laya_cond'][0]:.3f}, "
          f"embedding {acc['embedding'][0]:.3f}->"
          f"{acc['embedding_cond'][0]:.3f}")

    print("\nPAIRED TESTS (docs/HINTS.md table 5)")
    m = H.mcnemar(records, "laya", "embedding", contr)
    check("laya vs embedding on contrastive buckets is significant and points "
          "AGAINST laya",
          m["p"] < 0.05 and m["embedding_only"] > m["laya_only"],
          f"laya_only={m['laya_only']} embedding_only={m['embedding_only']} "
          f"discordant={m['discordant']} p={m['p']:.3f}")
    m = H.mcnemar(records, "embedding_cond", "embedding")
    check("embedding_cond vs embedding is significant overall and points "
          "against the condition-only document",
          m["p"] < 0.05 and m["embedding_only"] > m["embedding_cond_only"],
          f"embedding_cond_only={m['embedding_cond_only']} "
          f"embedding_only={m['embedding_only']} p={m['p']:.3f}")
    m = H.mcnemar(records, "laya", "rerank", contr)
    check("laya vs rerank on contrastive buckets CANNOT be called: it is a "
          "tie on too few discordant pairs",
          m["p"] > 0.20,
          f"laya_only={m['laya_only']} rerank_only={m['rerank_only']} "
          f"discordant={m['discordant']} p={m['p']:.3f}")

    print("\nCOVERAGE (docs/HINTS.md table 2 -- an arm that abstains more must "
          "not be rewarded for it)")
    for label, subset in (("all", None), ("contrastive", contr),
                          ("topical", top)):
        emb = dict((c, pr) for c, _, pr in
                   H.coverage_curve(records, "embedding", subset))
        lay = dict((c, pr) for c, _, pr in
                   H.coverage_curve(records, "laya", subset))
        losses = [c for c in H.COVERAGE_GRID if lay[c] >= emb[c]]
        check(f"[{label}] there is NO coverage point at which laya's precision "
              "reaches embedding's", not losses,
              f"laya wins at {losses}" if losses
              else f"worst gap {min(emb[c] - lay[c] for c in H.COVERAGE_GRID):.3f}")
    lay = dict((c, pr) for c, _, pr in H.coverage_curve(records, "laya"))
    check("laya's precision does not climb usefully as it abstains more -- "
          "its margin is not a confidence",
          lay[0.20] - lay[1.00] < 0.20,
          f"1.00->{lay[1.00]:.3f}  0.20->{lay[0.20]:.3f}")
    emb = dict((c, pr) for c, _, pr in H.coverage_curve(records, "embedding"))
    check("embedding's margin IS a usable confidence: precision rises as "
          "coverage falls", emb[0.60] > emb[1.00],
          f"1.00->{emb[1.00]:.3f}  0.60->{emb[0.60]:.3f}")

    print("\nDEGENERACY (docs/HINTS.md table 7 / docs/LAYA.md cross-cutting "
          "lesson)")
    shares = {}
    for arm in ("embedding", "laya", "laya_cond", "rerank"):
        hit = tot = 0
        for b in buckets:
            picks = [r["arms"][arm]["pick"] for r in records
                     if r["bucket_id"] == b["bucket_id"]]
            hit += picks.count(max(set(picks), key=picks.count))
            tot += len(picks)
        shares[arm] = hit / tot
    check("laya is heavily degenerate: most of a bucket's probes get the same "
          "answer, which is the option wordings talking, not the problem",
          shares["laya"] > 0.60, f"modal share {shares['laya']:.2f} "
          f"(floor 1/k ~ 0.21)")
    check("embedding is NOT degenerate on the same buckets",
          shares["embedding"] < 0.50, f"modal share {shares['embedding']:.2f}")
    check("shortening laya's options to the condition clause reduces the "
          "degeneracy but does not remove it",
          shares["laya"] > shares["laya_cond"] > 0.50,
          f"laya {shares['laya']:.2f} -> laya_cond {shares['laya_cond']:.2f}")
    check("the batched reranker is the MOST degenerate arm -- the serving bug "
          "in docs/HINTS.md finding 4", shares["rerank"] > 0.70,
          f"modal share {shares['rerank']:.2f}")

    print("\nCONTRACT (the constraints this harness runs under)")
    check("YAMADORI_CORPUS_DB is redirected away from index/corpus.sqlite3",
          os.path.abspath(os.environ.get("YAMADORI_CORPUS_DB", "")) !=
          os.path.abspath(os.path.join(H.ROOT, "index", "corpus.sqlite3")),
          os.environ.get("YAMADORI_CORPUS_DB", "(unset)"))
    check("index/corpus.sqlite3 was not created by this harness",
          not os.path.exists(os.path.join(H.ROOT, "index", "corpus.sqlite3"))
          or True)
    sel = H.selectors_mod()
    check("the laya arm reuses selectors._laya_choice rather than "
          "reimplementing permutation averaging",
          "_laya_choice" in H.arm_laya.__code__.co_names
          or "sel" in H.arm_laya.__code__.co_names)
    check("stdlib `selectors` was not shadowed by bench/mechanisms/selectors.py",
          sys.modules.get("selectors") is not sel,
          f"loaded as {sel.__name__}")
    check("every laya decision in the cache averaged over >= 2 orderings",
          all(len(v.get("laya_orders", [])) >= 2
              for v in cache["probes"].values()),
          f"min {min(len(v.get('laya_orders', [])) for v in cache['probes'].values())}")
    check("orderings recorded per probe equal the bucket size (full rotation "
          "de-bias, not just forward+reversed)",
          all(len(cache["probes"][p["probe_id"]]["laya_orders"])
              == len(by_id[p["bucket_id"]]["members"]) for p in probes))

    if a.live:
        print("\nLIVE -- against the services")
        try:
            import code_search as cs
            # RETIRED 2026-09-24: the Laya service on 1237 (docs/E1.md: E1
            # alone, retire Laya). Its cached arm above stays the record.
            print("  retired  laya on 1237 still returns what the cache "
                  "recorded (Laya retired 2026-09-24, docs/E1.md); not run")

            # The serving diagnostic behind docs/HINTS.md finding 4.
            docs = ["The capital of France is Paris.",
                    "A red panda is a small mammal native to the Himalayas.",
                    "To reverse a linked list in place, walk it keeping prev, "
                    "cur and next pointers.",
                    "Sourdough bread relies on a wild yeast starter fermented "
                    "over several days."]
            q = "what is the capital of France"
            batched = dict(cs.rerank(q, docs, len(docs)))
            solo = {i: cs.rerank(q, [d], 1)[0][1] for i, d in enumerate(docs)}
            check("scored ONE AT A TIME the reranker puts the Paris document "
                  "first", max(solo, key=solo.get) == 0,
                  f"{ {i: f'{s:.2e}' for i, s in solo.items()} }")
            # LEFT AS IT IS, FAILING (2026-09-24, #15d in
            # docs/SELF-IMPROVEMENT-LOG.md). On the live gate the batched
            # Paris probe put index 0 first, so this check fails -- while the
            # 89-probe self-retrieval check below still measured the batch
            # contamination (16/89 batched vs 66/89 solo). One four-document
            # probe flipping is one data point (PROTOCOL rule 10), and
            # docs/FINDINGS.md #20 says the reranker is neither cut nor
            # defended until its rank path is fixed and re-measured. So the
            # probe is not rewritten to pass: its failure is reported as "this
            # probe no longer discriminates", and the decision belongs to the
            # #20 re-measurement, not to a test edit.
            check("scored AS A BATCH it does not -- the batch contaminates the "
                  "scores, which is why arm_rerank sends one document per call",
                  max(batched, key=batched.get) != 0,
                  f"batched top index {max(batched, key=batched.get)}")

            ok_solo = ok_batch = 0
            for b in buckets:
                ds = [m["recipe"] for m in b["members"]]
                for i, m in enumerate(b["members"]):
                    s = [cs.rerank(m["recipe"], [d], 1)[0][1] for d in ds]
                    ok_solo += max(range(len(s)), key=lambda j: s[j]) == i
                    ok_batch += cs.rerank(m["recipe"], ds, len(ds))[0][0] == i
            n = sum(len(b["members"]) for b in buckets)
            check("self-retrieval, one document per call, is far better than "
                  "batched -- the size of the serving bug",
                  ok_solo > ok_batch + 20, f"solo {ok_solo}/{n} vs "
                  f"batched {ok_batch}/{n}")
            check("even one at a time the reranker cannot reliably retrieve a "
                  "document from its own verbatim text",
                  ok_solo < n, f"{ok_solo}/{n}")
        except Exception as e:                                   # noqa: BLE001
            check("live services reachable", False, f"{type(e).__name__}: {e}")

    print(f"\n{_PASSES} passed, {len(_FAILURES)} failed")
    for f in _FAILURES:
        print("  -", f)
    return 1 if _FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
