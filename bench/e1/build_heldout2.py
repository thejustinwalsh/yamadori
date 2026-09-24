#!/usr/bin/env python
"""Build the CANDIDATES for route_in held-out set 2 (docs/E1.md). Text only.

    PY=C:/Users/jwals/textgen/installer_files/env/python.exe
    $PY bench/e1/build_heldout2.py

Writes index/e1/heldout2_candidates.jsonl: {id, key, source, question,
context}. index/ is gitignored and the repo is PUBLIC: request text from the
corpus never enters the tree. No model is called and no prediction is written
here: the candidates are labelled BY HAND against RUBRIC (below) into
bench/e1/route_heldout2_labels.jsonl -- ids, keys, labels and reasons, no
text -- before any arm is scored on them (docs/CLM-EVAL.md s.5, "labelled
blind"). `key` is sha256(question NUL context)[:12]; e1.heldout2_rows()
joins the two files on id and refuses a key that does not match.

SOURCES, in this order (the corpus first -- real traffic through :1234):

  corpus_real       every distinct user-speaking turn in index/corpus.sqlite3
                    (Hermes, the operator's own dogfood sessions, the live
                    test suites' sessions), minus the exclusions below
  corpus_benchmark  benchmark prompts that reached the corpus (LiveCodeBench,
                    SWE-bench PR descriptions, the Guardian summaries):
                    a seeded sample, capped, so they do not swamp the set
  bench_other       top-up from bench files NO route_in training or held-out
                    file matches: bench/context_economy_tasks.jsonl (three.js
                    source questions) and a seeded sample of
                    bench/hint_probes.jsonl (algorithm problems phrased as
                    "our ..." -- surface-cue negatives for the regex)

EXCLUDED (not route_in questions, or not ours to label):
  - client side calls: chat-title namers, security-review verdicts,
    compaction / summarisation prompts, memory and skill review passes
  - harness notices and bare continuations ("continue", "[System: ...]")
  - requests for sexual imagery of real people (present in the corpus;
    not a routing question, not labelled, not embedded)
  - PRIVACY (operator rule, 2026-09-24): any image-generation request, any
    turn carrying an image prompt, an image or any attachment (@file,
    @folder, "Attached Context", pasted-content blocks). index/media is
    never opened. Docs cite ids and counts only, never user text.

DEDUP: a candidate is dropped when its normalised word set has Jaccard >=
NEAR_DUP with any route_in training row (bench/laya_routing_labels*.jsonl),
the first held-out set (bench/laya_routing_heldout_packages.jsonl), or a
candidate already kept.
"""
from __future__ import annotations

import glob
import json
import os
import random
import re
import sqlite3
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "mcp"))

OUT = os.path.join(ROOT, "index", "e1", "heldout2_candidates.jsonl")
CORPUS = os.path.join(ROOT, "index", "corpus.sqlite3")
NEAR_DUP = 0.7
SEED = 20260924
CAPS = {"lcb": 12, "swe": 4, "guardian": 3, "hint_probes": 24}

# THE RUBRIC the labels are written against. Its wording is the route_in
# option descriptions Laya's head was trained with (mcp/laya_head.TASKS),
# extended to name the held library versions -- which is how the first
# held-out set's `why` fields read ("must be read from the r185 source").
RUBRIC_VERSION = "route_in-rubric-v1 (2026-09-24)"
RUBRIC = {
    "investigate": "answering correctly requires reading source the "
                   "assistant can open and the conversation does not "
                   "contain: this repository's own files, or a held library "
                   "version's source/exports (a version-specific API, a "
                   "default, where something is defined, whether it still "
                   "exists) -- a fact memory is likely to get wrong",
    "answer_directly": "general knowledge about a language, library, API or "
                       "technique, stable across versions; or everything "
                       "needed is already in the conversation (pasted code, "
                       "a spec); or a task done from scratch with no source "
                       "to read",
    "clarify": "too vague or underspecified to act on: it does not say what "
               "'it' is, what is wrong, or what outcome is wanted",
}

_SIDE = re.compile(r"You name chat sessions|security reviewer for an AI",
                   re.I)
_SIDE_REQ = re.compile(
    r"^(?:You are a summarization agent|Your task is to create a detailed "
    r"summary|Review the conversation above|\[CONTEXT COMPACTION|"
    r"\[System: |\[Note: model was just switched|continue\s*$)", re.I)
# Generic adult-content exclusion (private traffic is never test material).
_EXPLICIT = re.compile(r"\b(?:nude|naked|nsfw|explicit|sexual|erotic)\b", re.I)
_PRIVATE = re.compile(
    r"generate_image|describe_image|\bgenerate (?:an? |me )?(?:\w+ )?image|"
    r"\bdraw (?:a|an|me)\b|attached image|\bimage\b|\bpicture\b|"
    r"@file:|@folder:|--- Attached Context ---|Pasted content", re.I)
_INTERRUPTED = "[Context from the interrupted assistant response]"
# Follow-ups whose earlier user turn is known from the suite that sent them
# (the corpus stores each request, not the conversation). Anything else gets
# the previous request of the same session only when it is the same system
# prompt, within FOLLOW_UP_SECONDS.
KNOWN_CONTEXT = {
    "What is the first thing I should check? One sentence.":
        "The nightly ledger rollup job failed. Find out why.",
}
FOLLOW_UP_SECONDS = 120


def _key(question: str, context: str) -> str:
    import hashlib
    return hashlib.sha256((question + "\x00" + (context or "")).encode()
                          ).hexdigest()[:12]


def _words(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_]+", (s or "").lower()))


def _jac(a: set, b: set) -> float:
    return len(a & b) / max(1, len(a | b))


def _category(r: str) -> str:
    if "competitive programming" in r or r.startswith("### Instructions"):
        return "lcb"
    if "<pr_description>" in r:
        return "swe"
    if "news article from the Guardian" in r:
        return "guardian"
    return "other"


def _instruction(text: str) -> tuple[str, str]:
    """(the user's instruction, the attachment) -- selection's own split, so
    the question is what the live second signal would be handed."""
    import selection
    q, att = selection.instruction_of(text)
    return q.strip(), att


def corpus_turns() -> list[dict]:
    con = sqlite3.connect(f"file:{CORPUS}?mode=ro", uri=True)
    out, seen = [], set()
    prev_by_sys: dict[str, tuple[float, str]] = {}
    for ts, p in con.execute("SELECT ts, payload FROM events WHERE "
                             "kind='turn' ORDER BY ts"):
        d = json.loads(p)
        req = d.get("request") or ""
        sysh = d.get("system_head") or ""
        if req in seen:
            continue
        seen.add(req)
        last = prev_by_sys.get(sysh[:80])
        prev = (last[1] if last and not d.get("first_turn")
                and ts - last[0] <= FOLLOW_UP_SECONDS else None)
        prev_by_sys[sysh[:80]] = (ts, req)
        out.append({"ts": ts, "request": req, "system_head": sysh,
                    "utility": bool(d.get("utility")),
                    "first_turn": bool(d.get("first_turn")),
                    "prev": prev})
    con.close()
    return out


def main() -> None:
    import route
    ref = []
    for path in sorted(glob.glob(os.path.join(ROOT, "bench",
                                              "laya_routing_labels*.jsonl"))
                       + [os.path.join(ROOT, "bench",
                                       "laya_routing_heldout_packages.jsonl")]):
        with open(path, encoding="utf-8") as fh:
            for ln in fh:
                if ln.strip():
                    ref.append(_words(json.loads(ln)["question"]))
    kept: list[dict] = []
    kept_words: list[set] = []
    dropped = {"side_call": 0, "notice": 0, "explicit": 0,
               "image_or_attachment": 0, "near_dup": 0, "too_short": 0}

    def keep(src: str, q: str, ctx: str = "", note: str = "") -> bool:
        w = _words(q)
        if len(q.strip()) < 2:
            dropped["too_short"] += 1
            return False
        if any(_jac(w, r) >= NEAR_DUP for r in ref) or \
                any(_jac(w, k) >= NEAR_DUP for k in kept_words):
            dropped["near_dup"] += 1
            return False
        kept.append({"id": f"h2-{len(kept):03d}",
                     "key": _key(q[:2000], ctx[:1500]), "source": src,
                     "question": q[:2000], "context": ctx[:1500],
                     **({"note": note} if note else {})})
        kept_words.append(w)
        return True

    bench_rows: dict[str, list[str]] = {"lcb": [], "swe": [], "guardian": []}
    for t in corpus_turns():
        req = t["request"]
        cat = _category(req)
        if cat != "other":
            bench_rows[cat].append(req)
            continue
        if t["utility"] or _SIDE.search(t["system_head"]):
            dropped["side_call"] += 1
            continue
        body = req
        if body.startswith(_INTERRUPTED):
            # A user correction typed over an interrupted answer: the user's
            # own words are the last paragraph.
            body = body.split("\n\n")[-1]
        if _SIDE_REQ.search(body.strip()) or route.harness_notice(body):
            dropped["notice"] += 1
            continue
        if _EXPLICIT.search(body):
            dropped["explicit"] += 1
            continue
        if _PRIVATE.search(req):
            dropped["image_or_attachment"] += 1
            continue
        q, _att = _instruction(body)
        ctx = KNOWN_CONTEXT.get(q, "")
        if not ctx and not t["first_turn"] and t["prev"] and len(q) < 80:
            pq, _ = _instruction(t["prev"])
            if not _SIDE_REQ.search(pq) and not _EXPLICIT.search(pq):
                ctx = pq[:1500]
        keep("corpus_real", q, ctx)
    rnd = random.Random(SEED)
    for cat in ("lcb", "swe", "guardian"):
        rows = sorted(bench_rows[cat])
        rnd.shuffle(rows)
        for req in rows[:CAPS[cat]]:
            q, _ = _instruction(req)
            keep("corpus_benchmark", q, note=cat)
    with open(os.path.join(ROOT, "bench", "context_economy_tasks.jsonl"),
              encoding="utf-8") as fh:
        for ln in fh:
            if ln.strip():
                keep("bench_other", json.loads(ln)["question"],
                     note="context_economy")
    with open(os.path.join(ROOT, "bench", "hint_probes.jsonl"),
              encoding="utf-8") as fh:
        probes = [json.loads(ln) for ln in fh if ln.strip()]
    probes.sort(key=lambda p: p["probe_id"])
    rnd.shuffle(probes)
    n = 0
    for p in probes:
        if n >= CAPS["hint_probes"]:
            break
        n += keep("bench_other", p["problem"], note="hint_probes")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        for r in kept:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    by = {}
    for r in kept:
        by[r["source"]] = by.get(r["source"], 0) + 1
    print(f"{len(kept)} candidates -> {OUT}\n  by source {by}\n  dropped {dropped}")


if __name__ == "__main__":
    main()
