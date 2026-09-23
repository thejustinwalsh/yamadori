#!/usr/bin/env python
"""Attention boosters: recall the few skills that apply, and never harm.

THE THESIS THIS IMPLEMENTS

A corpus of small, specific engineering skills. At code-writing time, recall
the handful that apply to THIS request and put them in front of the model.
Memory recall strengthening the logic system -- the second half of the
dual-hemisphere bet, where fan-out is the creativity half.

THE CONSTRAINT THAT SHAPES EVERYTHING HERE

**A hint must never harm.** Emitting nothing is a perfectly good outcome; a
wrong attention-booster is an active harm, because it points the model at the
wrong thing precisely while it writes. So this is a PRECISION problem with
optional coverage, not an accuracy problem. Abstention is the feature.

That inverts the usual metric. Do not ask "how often is the top hint right".
Ask "when we speak, how often are we right, and can we stay quiet otherwise".

WHY EMBEDDINGS AND NOT THE DECISION MODEL

Measured, 19 buckets of mutually exclusive hints, 89 probes, chance 21.3%
(`docs/HINTS.md`):

    embedding                71.9%   (contrastive 57.4, topical 88.1)
    lexical floor, no model  51.7%
    laya, joint scoring      33.7%   (contrastive 29.8 -- worst where it was
                                      predicted to be best)
    rerank                   22.5%
    random / fixed           21.3%

Embeddings win everywhere, and -- the part that matters here -- their margin
is a real confidence: 71.9% -> 86.8% at 60% coverage, and 100% on topical
buckets from 75% coverage down. That is exactly the never-harm dial, already
calibrated, needing nothing trained.

The reranker is NOT used. Its scores depend on what else is in the same
request -- the same 89 documents score 15-16/89 batched against 68/89 one at a
time -- so it is corrupt as deployed. See FINDINGS #20.

WHERE THE HINTS GO, AND WHY IT IS THE USER MESSAGE

Not the system block. Two independent reasons:

  KV cache      the system prefix must stay byte-identical between turns or
                every turn pays a full prefill -- ~51s at 65k on this
                hardware. Hints are per-request, so they cannot live there.
  it works      `concept_seed.py` records the measurement: in the system
                message "the model sometimes ignored it and fell back to its
                default template", in the user message "it couldn't".

So hints ride with the request, appended to the last user turn, where they
are both effective and cache-neutral.

BUCKETS: HINTS THAT WOULD ARGUE ARE COLLAPSED TO ONE

Demonstrated live (docs/HANDOFF.md): "static array, many range-sum queries"
returned "Static array ...: prefix sums" AND "Point updates interleaved with
range sums: Fenwick tree" -- two mutually exclusive answers, side by side, in
front of a model that then has to referee them. `bench/hint_buckets.jsonl`
groups such alternatives (19 buckets, 89 members, each with `why_exclusive`,
keyed on "what question does this answer", not on `area`/`category`).

When two or more candidates above the floor share a bucket, only the one with
the highest cosine survives. That is the measured-best collapse strategy:
`bench/hint_collapse.py` (TABLE 1, n=89 probes, paired, cached in
`bench/data/hint_collapse_runs.json`): embedding argmax 64/89 = 71.9%
(contrastive 27/47 = 57.4%, topical 37/42 = 88.1%) against lexical 51.7%,
laya 33.7%, rerank 22.5%, chance 21.3%. Greedy "walk by descending cosine,
skip a bucket already represented" IS that argmax -- the first member of a
bucket reached is the bucket's highest-scoring member.

Rows in no bucket behave exactly as before. The freed slot goes to the next
candidate that clears the same floor, so the output never depends on whether a
losing sibling happens to exist -- and nothing below the floor is ever used to
fill it. Contrastive buckets are still the weak case (57.4%): collapsing
replaces "two hints that argue" with "one hint that is wrong 4 times in 10" on
those buckets.

On the PRODUCTION path (`python mcp/hints.py --probes`: real index/hints.npz,
real embedder, all 89 probes, k=3, floor 0.55; run twice 2026-09-22,
identical both times):

    outputs where 2+ siblings argue       2 -> 0
    correct member shown                 30 -> 30   (lost to collapse: 0)
    a wrong sibling shown                10 -> 8
    correct and ONLY correct shown       28 -> 30
    no member of the probe's bucket      51 -> 51   (the floor abstaining)
    bucket argmax, production vectors    67/89 vs the harness's 64/89
                                         (5 vs 2 discordant, p=0.453: the
                                         `_hint_text` document does not change
                                         the strategy's standing at this n)

Only 2 of 89 probes hit the arguing case, so "2 -> 0" is a property of the
construction, not a statistical result; the probes avoid the hints' own
vocabulary and mostly land below the floor.

Coverage is the open problem: 89 of 938 recipes are bucketed. An unbucketed
pair of alternatives still argues.
"""
from __future__ import annotations

import glob
import json
import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.environ.get(
    "YAMADORI_HINT_CORPUS", os.path.join(HERE, "..", "bench", "recipes"))
CACHE = os.environ.get(
    "YAMADORI_HINT_CACHE", os.path.join(HERE, "..", "index", "hints.npz"))
# Mutually exclusive alternatives. Not part of the cache signature: buckets
# change which vectors compete, never what a vector is, so editing this file
# must not cost a re-embed.
BUCKETS = os.environ.get(
    "YAMADORI_HINT_BUCKETS",
    os.path.join(HERE, "..", "bench", "hint_buckets.jsonl"))

# How many hints may reach the model. Small on purpose: this is an attention
# booster, and a wall of advice is the opposite of focused attention.
TOP_K = int(os.environ.get("YAMADORI_HINT_K", "3"))

# THE NEVER-HARM DIAL, and it is the only number in this file that decides
# whether a hint is emitted.
#
# Cosine must clear this for a hint to be sent at all. Set from the measured
# coverage/precision curve in docs/HINTS.md: embeddings reach 86.8% precision
# at 60% coverage. It is deliberately a similarity floor rather than a rank
# cutoff -- "the best of a bad set" is exactly the case where staying quiet is
# correct, and a rank cutoff cannot express that.
#
# UNCALIBRATED FOR THIS EXACT USE. The curve was measured on bucket collapse
# (choose among known alternatives), not on open retrieval over 938 recipes.
# Treat it as a starting point and re-measure; see docs/ROADMAP.md 1.5.
FLOOR = float(os.environ.get("YAMADORI_HINT_FLOOR", "0.55"))

# The review state that removes a row from the corpus. Spelled as dashboard.py
# writes it (its CHIP map: keep / edited / reject / unreviewed).
REJECTED = "reject"

_LOCK = threading.Lock()
_ROWS: list[dict] = []
_MAT = None
_BUCKET_STATUS: dict = {}


def _load_corpus() -> list[dict]:
    rows = []
    for path in sorted(glob.glob(os.path.join(CORPUS, "*.jsonl"))):
        stem = os.path.basename(path)
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                # Review is optional and after the fact, so an unreviewed row
                # is served. A row a person REJECTED is not: that is the one
                # review decision that must reach the model.
                if r.get("recipe") and r.get("_state") != REJECTED:
                    r["_file"] = stem
                    rows.append(r)
    return rows


def _assign_buckets(rows: list[dict], path: str = None) -> dict:
    """Stamp `_bucket` on every row that is a bucket member. Returns a status.

    Members are matched on VERBATIM recipe text, the same key
    `bench/hint_collapse.py:verify_no_drift` checks. Not on `ref`
    ("complexity:21"): that is a line number, and a row inserted above it
    would silently move a bucket onto an unrelated recipe. Text drift fails the
    other way -- an edited recipe drops out of its bucket and is served
    unbucketed, i.e. today's behaviour -- and it is reported, not swallowed:
    an unmatched member is exactly how two arguing hints come back.
    """
    path = path or BUCKETS
    status = {"path": path, "buckets": 0, "members": 0, "matched": 0,
              "rows_bucketed": 0, "unmatched": [], "error": None}
    by_text: dict[str, str] = {}
    try:
        with open(path, encoding="utf-8") as fh:
            for n, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    b = json.loads(line)
                    bid = b["bucket_id"]
                    members = b["members"]
                except (ValueError, KeyError, TypeError) as e:
                    status["error"] = (f"line {n} unreadable "
                                       f"({type(e).__name__}); skipped")
                    continue
                status["buckets"] += 1
                for m in members:
                    text = (m.get("recipe") or "").strip()
                    if text:
                        status["members"] += 1
                        by_text[text] = bid
    except OSError as e:
        status["error"] = f"{type(e).__name__}: {e}"

    seen: set[str] = set()
    for r in rows:
        bid = by_text.get((r.get("recipe") or "").strip())
        if bid is not None:
            r["_bucket"] = bid
            status["rows_bucketed"] += 1
            seen.add(r["recipe"].strip())
    status["matched"] = len(seen)
    status["unmatched"] = sorted(f"{b}: {t[:60]}" for t, b in by_text.items()
                                 if t not in seen)
    if status["error"] or status["unmatched"]:
        print(f"  hints: buckets degraded ({path}): "
              f"{status['matched']}/{status['members']} members matched"
              + (f"; {status['error']}" if status["error"] else "")
              + (f"; unmatched (edited upstream or rejected in review): "
                 f"{status['unmatched'][:3]}" if status["unmatched"] else "")
              + " -- unmatched alternatives are served unbucketed and can "
                "argue", file=sys.stderr, flush=True)
    return status


def bucket_status() -> dict:
    """How much of the bucket file reached the served corpus. Empty before
    the first load()."""
    return dict(_BUCKET_STATUS)


def _hint_text(r: dict) -> str:
    """What gets EMBEDDED for a recipe.

    The whole recipe, not just its trigger condition. Measured: embedding the
    `trigger_condition` alone costs embeddings 14.6 points (p=0.015) -- cosine
    needs the whole passage. It is the opposite of what helps the decision
    model, which is one reason the two mechanisms cannot share a
    representation. See docs/HINTS.md.
    """
    bits = [r.get("recipe") or ""]
    for key in ("trigger_condition", "area", "category", "language"):
        v = r.get(key)
        if isinstance(v, str) and v:
            bits.append(v)
    return "\n".join(bits)


def ready() -> bool:
    return _MAT is not None and len(_ROWS) > 0


def load(force: bool = False):
    """Load the corpus and its vectors, building the cache on first use.

    Building needs the embedding server. Loading a built cache does not, which
    is what keeps this off the hot path and testable without a GPU.
    """
    global _ROWS, _MAT, _BUCKET_STATUS
    import numpy as np

    with _LOCK:
        if ready() and not force:
            return _ROWS, _MAT
        rows = _load_corpus()
        if not rows:
            return [], None
        texts = [_hint_text(r) for r in rows]

        cached = None
        if os.path.exists(CACHE) and not force:
            try:
                z = np.load(CACHE, allow_pickle=False)
                if int(z["n"]) == len(texts) and str(z["sig"]) == _signature(texts):
                    cached = z["mat"]
            except Exception:                                    # noqa: BLE001
                cached = None

        if cached is None:
            import code_search as cs
            # Documents take NO instruction prefix -- Qwen3-Embedding is
            # asymmetric, and getting this backwards does not error, it
            # silently returns near-random neighbours.
            vecs = []
            for i in range(0, len(texts), 64):
                vecs.append(cs.embed(texts[i:i + 64], is_query=False))
            cached = np.vstack(vecs)
            os.makedirs(os.path.dirname(os.path.abspath(CACHE)), exist_ok=True)
            np.savez(CACHE, mat=cached, n=len(texts), sig=_signature(texts))

        # After the cache, and outside its signature: the bucket is stamped on
        # the row itself so it cannot drift from the row across a refresh().
        _BUCKET_STATUS = _assign_buckets(rows)
        _ROWS, _MAT = rows, cached
        return _ROWS, _MAT


def refresh():
    """Re-read the corpus from disk, re-embedding only if it changed.

    `load()` returns the in-memory copy once it has one, so a long-lived
    process (the worker, the proxy) never sees rows written after its first
    call. This drops that copy; the on-disk cache is still reused whenever its
    signature matches, so an unchanged corpus costs a file read, not a pass
    over the embedding server.
    """
    global _ROWS, _MAT
    with _LOCK:
        _ROWS, _MAT = [], None
    return load()


def _signature(texts: list[str]) -> str:
    import hashlib
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode("utf-8", "replace"))
    return h.hexdigest()[:16]


def task_domains(messages: list[dict]) -> set[str] | None:
    """The task's domains for the eligibility filter, or None for NO filter.

    The filter RESTRICTS, so it runs only on strong evidence -- an import or a
    file extension, a fact about the code (`domains.strong_evidence`). Words
    then add to that set (`domains.detect`), as the domains module intends:
    words may add a domain, never remove one. A task with no strong evidence
    is not filtered at all.

    WHY NOT `eligible(detect(messages))` AS WRITTEN IN THE PLAN. Measured
    offline on the 89 hint probes, argmax over the cached harness scores
    (bench/data/hint_collapse_runs.json; unfiltered 64/89):

        filter on detect(), strict corpora closed on no evidence   41/89
        filter on detect() only when it found something             59/89
        filter only on strong evidence (this)                       64/89

    That last row is 64 BY CONSTRUCTION: no probe carries an import or a file
    extension, so none is filtered. The probes cannot measure this filter's
    benefit, only prove it costs them nothing; mcp/test_hints.py tests the
    filter itself.

    The probes are prose written to avoid the hints' own vocabulary, so word
    detection either finds nothing -- and the strict visual-design corpus,
    closed on no evidence, lost all 27 of its probes -- or finds the wrong
    thing ("backend" from a Zig allocator question). A filter that removes
    the right hint on the probe set fails the never-harm rule on it. Before
    this there was no filter at all, so this is never looser than before.
    """
    import discover
    import domains
    try:
        imps = discover.imports(domains._blob(messages))
        if not domains.strong_evidence(messages, imports=imps):
            return None
        return domains.detect(messages, imports=imps)
    except Exception:                                            # noqa: BLE001
        # Detection is a filter on an enhancement. If it fails, hints go out
        # unfiltered, exactly as before the filter existed, rather than the
        # failure taking the hints down with it.
        return None


def select(question: str, k: int = TOP_K, floor: float = FLOOR,
           collapse: bool = True, task: set[str] | None = None) -> list[dict]:
    """The hints worth showing, or an empty list.

    Empty is a correct and common answer. Nothing here tries to fill k.
    `collapse=False` is today's-minus-buckets behaviour, kept so the two can
    be compared on identical vectors (`--probes`), not as a serving option.

    DOMAIN FILTER FIRST (docs/SELECTION-BUILD.md step 6). A recipe whose
    domains do not meet the task's is not ranked low, it is ABSENT: a cosine
    says how similar two texts are, not whether a rule is admissible
    (mcp/domains.py). `task` is the task's domains; None detects them from
    `question` alone, and there is no filter without strong evidence -- see
    `task_domains`. `domains.eligible` and `filter_recipes` existed with no
    callers until this.
    """
    rows, mat = load()
    if mat is None or not question.strip():
        return []
    if task is None:
        task = task_domains([{"role": "user", "content": question}])
    import code_search as cs
    # Queries DO take the instruction prefix. See the note in _load.
    q = cs.embed([question], is_query=True)[0]
    return _select_vector(rows, mat, q, k, floor, collapse, task)


def _eligible(row: dict, task: set[str] | None) -> bool:
    if task is None:
        return True
    import domains
    return domains.eligible(row, task, row.get("_file", ""))


def _select_vector(rows, mat, q, k: int, floor: float,
                   collapse: bool = True,
                   task: set[str] | None = None) -> list[dict]:
    """select() after the embedding call. Split out so a measurement can
    score both arms on ONE query vector instead of two embed round trips.

    `task` None is NO domain filter -- for a measurement that wants the
    unfiltered ranking; serving always passes a set (possibly empty)."""
    import numpy as np

    sims = mat @ q
    # Stable, so equal cosines resolve by corpus order on every run.
    order = np.argsort(-sims, kind="stable")
    out: list[dict] = []
    winner: dict[str, dict] = {}         # bucket_id -> the hint that holds it
    for i in order:
        s = float(sims[i])
        if s < floor:
            break            # sorted, so everything after is worse
        row = rows[int(i)]
        if not _eligible(row, task):
            # Inadmissible for this task: skipped BEFORE the bucket collapse,
            # so an ineligible sibling can neither hold a bucket nor take a
            # slot. The next eligible candidate above the floor gets it.
            continue
        bid = row.get("_bucket") if collapse else None
        if bid is not None and bid in winner:
            # A sibling with a higher cosine already holds this bucket: this
            # is the argmax collapse, measured best at 71.9% (n=89). Kept on
            # the winner so a log can show what was withheld and why.
            winner[bid]["_suppressed"].append(
                {"score": round(s, 4), "recipe": row["recipe"][:120]})
            continue
        if len(out) >= max(k, 1):
            break
        r = dict(row)
        r["_score"] = round(s, 4)
        if bid is not None:
            r["_suppressed"] = []
            winner[bid] = r
        out.append(r)
    return out


def render(hints: list[dict]) -> str:
    """The block appended to the user's turn. Empty when there is nothing."""
    if not hints:
        return ""
    lines = ["", "---",
             "Relevant engineering notes from this server's corpus. They are "
             "suggestions, not requirements -- if one does not fit this "
             "problem, ignore it and say nothing about it."]
    for h in hints:
        src = h.get("source_name") or h.get("_file", "")
        lines.append(f"- {h['recipe'].strip()}" + (f"  [{src}]" if src else ""))
    return "\n".join(lines)


def attach(messages: list[dict], k: int = TOP_K,
           floor: float = FLOOR) -> tuple[list[dict], list[dict]]:
    """Append hints to the last user turn. Returns (messages, hints_used).

    The LAST USER TURN, not the system block: the system prefix must stay
    byte-identical or every turn pays a full prefill, and the system message is
    also where this model has been measured to ignore injected guidance.
    """
    out = [dict(m) for m in messages]
    idx = next((i for i in range(len(out) - 1, -1, -1)
                if out[i].get("role") == "user"), None)
    if idx is None:
        return out, []
    content = out[idx].get("content")
    if not isinstance(content, str) or not content.strip():
        return out, []
    # The WHOLE conversation's domains, not only the last turn's: an import
    # pasted two turns ago is still the task's evidence, and more evidence
    # can only add domains -- detect() never removes one.
    hints = select(content, k, floor, task=task_domains(messages))
    block = render(hints)
    if block:
        out[idx]["content"] = content + "\n" + block
    return out, hints


def _mcnemar_p(b: int, c: int) -> float:
    """Exact two-sided McNemar on the discordant counts."""
    from math import comb
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(0, min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def evaluate_probes(probes_path: str = None) -> dict:
    """Bucket collapse on the PRODUCTION path, over bench/hint_probes.jsonl.

    Real cache, real embedding server, real select(): one query vector per
    probe, scored with collapse off and on. What it answers that
    bench/hint_collapse.py does not: that harness embeds the bare recipe and
    ranks within the bucket only; production embeds `_hint_text` (recipe +
    trigger + area + category + language) and ranks against all 938 rows under
    a floor. The strategy was chosen there; whether it survives here is this.
    """
    import code_search as cs
    root =os.path.join(HERE, "..", "bench")
    probes_path = probes_path or os.path.join(root, "hint_probes.jsonl")
    probes = [json.loads(x) for x in open(probes_path, encoding="utf-8")
              if x.strip()]
    buckets = {}
    for x in open(BUCKETS, encoding="utf-8"):
        if x.strip():
            b = json.loads(x)
            buckets[b["bucket_id"]] = {m["member_id"]: m["recipe"].strip()
                                       for m in b["members"]}
    cached = {}
    runs = os.path.join(root, "data", "hint_collapse_runs.json")
    if os.path.exists(runs):
        with open(runs, encoding="utf-8") as fh:
            cached = json.load(fh).get("probes", {})

    rows, mat = load()
    index = {r["recipe"].strip(): i for i, r in enumerate(rows)}
    qv = cs.embed([p["problem"] for p in probes], is_query=True)
    per = []
    for p, q in zip(probes, qv):
        members = buckets[p["bucket_id"]]
        right = members[p["correct"]]
        # The production filter applies here too, or this would measure a
        # ranking production no longer serves: a member the probe's domains
        # make inadmissible cannot be picked.
        task = task_domains([{"role": "user", "content": p["problem"]}])
        present = {mid: index[t] for mid, t in members.items()
                   if t in index and _eligible(rows[index[t]], task)}
        sims = mat @ q
        prod_pick = (max(present, key=lambda mid: float(sims[present[mid]]))
                     if present else None)
        emb = (cached.get(p["probe_id"]) or {}).get("embedding")
        row = {"probe": p["probe_id"], "type": p["bucket_type"],
               "prod_argmax_ok": prod_pick == p["correct"],
               "harness_argmax_ok": (max(emb, key=emb.get) == p["correct"]
                                     if emb else None)}
        for arm, on in (("before", False), ("after", True)):
            got = _select_vector(rows, mat, q, TOP_K, FLOOR, collapse=on,
                                 task=task)
            texts = [h["recipe"].strip() for h in got]
            mine = [t for t in texts if t in members.values()]
            counts: dict[str, int] = {}
            for h in got:
                if h.get("_bucket"):
                    counts[h["_bucket"]] = counts.get(h["_bucket"], 0) + 1
            row[arm] = {"n": len(got),
                        "argues": any(v > 1 for v in counts.values()),
                        "correct": right in texts,
                        "wrong_sibling": any(t != right for t in mine),
                        "only_correct": mine == [right],
                        "bucket_silent": not mine}
        per.append(row)

    def count(pred, subset=None):
        return sum(1 for r in per if (subset is None or r["type"] == subset)
                   and pred(r))

    out = {"n": len(per), "k": TOP_K, "floor": FLOOR, "rows": per}
    for arm in ("before", "after"):
        out[arm] = {key: count(lambda r, a=arm, kk=key: r[a][kk])
                    for key in ("argues", "correct", "wrong_sibling",
                                "only_correct", "bucket_silent")}
        out[arm]["mean_hints"] = round(
            sum(r[arm]["n"] for r in per) / len(per), 2)
    out["prod_argmax"] = {s or "all": count(lambda r: r["prod_argmax_ok"], s)
                          for s in (None, "contrastive", "topical")}
    paired = [r for r in per if r["harness_argmax_ok"] is not None]
    b = sum(1 for r in paired if r["prod_argmax_ok"]
            and not r["harness_argmax_ok"])
    c = sum(1 for r in paired if r["harness_argmax_ok"]
            and not r["prod_argmax_ok"])
    out["prod_vs_harness"] = {"n": len(paired),
                              "harness_ok": sum(r["harness_argmax_ok"]
                                                for r in paired),
                              "prod_only": b, "harness_only": c,
                              "p": round(_mcnemar_p(b, c), 3)}
    lost = sum(1 for r in per if r["before"]["correct"]
               and not r["after"]["correct"])
    gained = sum(1 for r in per if r["after"]["correct"]
                 and not r["before"]["correct"])
    out["correct_lost_to_collapse"] = lost
    out["correct_gained"] = gained
    return out


def _print_probe_report(res: dict) -> None:
    n = res["n"]
    print(f"\n  PRODUCTION PATH, n={n} probes, k={res['k']}, "
          f"floor={res['floor']}\n")
    print(f"  {'':34}{'before':>10}{'after':>10}")
    labels = {"argues": "outputs where 2+ siblings argue",
              "correct": "correct member shown",
              "wrong_sibling": "a wrong sibling shown",
              "only_correct": "correct and ONLY correct shown",
              "bucket_silent": "no member of its bucket shown",
              "mean_hints": "mean hints per output"}
    for key, lab in labels.items():
        print(f"  {lab:34}{res['before'][key]:>10}{res['after'][key]:>10}")
    print(f"\n  correct member lost to collapse: "
          f"{res['correct_lost_to_collapse']}   gained: "
          f"{res['correct_gained']}")
    pa = res["prod_argmax"]
    print(f"  bucket argmax on PRODUCTION vectors: {pa['all']}/{n} "
          f"(contrastive {pa['contrastive']}, topical {pa['topical']})")
    ph = res["prod_vs_harness"]
    print(f"  vs harness recipe-only vectors: {ph['harness_ok']}/{ph['n']}; "
          f"discordant prod-only {ph['prod_only']} / harness-only "
          f"{ph['harness_only']}, McNemar p={ph['p']}")


if __name__ == "__main__":
    rows = _load_corpus()
    print(f"  {len(rows)} recipes across "
          f"{len({r['_file'] for r in rows})} files")
    print(f"  cache: {CACHE}  exists={os.path.exists(CACHE)}")
    print(f"  TOP_K={TOP_K}  FLOOR={FLOOR}")
    if sys.argv[1:2] == ["--probes"]:
        res = evaluate_probes(*sys.argv[2:3])
        st = bucket_status()
        print(f"  buckets: {st['matched']}/{st['members']} members matched, "
              f"{st['rows_bucketed']} rows bucketed")
        _print_probe_report(res)
        sys.exit(0)
    if len(sys.argv) > 1:
        q = " ".join(sys.argv[1:])
        hs = select(q)
        print(f"\n  query: {q!r}\n  {len(hs)} hint(s) above the floor:")
        for h in hs:
            tag = f"  [bucket {h['_bucket']}]" if h.get("_bucket") else ""
            print(f"    {h['_score']:.3f}  {h['recipe'][:96]}{tag}")
            for s in h.get("_suppressed", []):
                print(f"        withheld {s['score']:.3f}  {s['recipe'][:80]}")
        if not hs:
            print("    (silent -- nothing cleared the floor, which is a "
                  "correct outcome)")
