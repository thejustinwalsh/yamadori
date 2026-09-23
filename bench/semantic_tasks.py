#!/usr/bin/env python
"""Natural-language retrieval tasks, generated and then verified against leakage.

WHY THIS EXISTS SEPARATELY

The locate benchmark asks "where is `createBindGroupLayout` defined?" and
feeds the bare symbol to the retriever. That is a fair test of symbol lookup
and an UNFAIR test of a reranker: a cross-encoder is trained on natural
language queries paired with documents, and a one-word identifier is a
degenerate input for it. Measured at n=237 it scored hit@1 13.9% against the
embedding ranking's 80.2%, which is a real result about that configuration and
says nothing about reranking as such.

This generates the input a reranker is actually for: a question in prose, with
a known correct file.

THE LEAKAGE PROBLEM, AND THE FIX

The obvious source of prose is the doc comment above each definition. It is
also indexed, as part of the chunk it documents, so using it verbatim as the
query means asking the retriever to find a document that CONTAINS the query.
Embedding search would score near 1.0 and the benchmark would measure string
identity while looking like it measured meaning.

So the local model paraphrases each doc comment into a question, and the
paraphrase is kept only if it shares few enough content words with the source
-- the same verified-loop shape as `make_ste_pairs.py`, where a mechanical
check decides and nothing is trusted for sounding right. `MAX_OVERLAP` is the
knob; the rejection counts are printed so the reader can see how much was
thrown away and why.

A question that survives is one whose answer requires matching meaning rather
than matching tokens, which is the only kind that can tell an embedding and a
reranker apart.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sqlite3
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "mcp"))

UPSTREAM = os.environ.get("LLAMA_STACK_URL", "http://127.0.0.1:11434")
MODEL = os.environ.get("BENCH_MODEL", "bonsai")

# A doc comment, JSDoc or block style, at the head of a chunk.
_DOC = re.compile(r"/\*\*(.+?)\*/", re.S)
_TAG = re.compile(r"^\s*@\w+.*$", re.M)

# Above this share of the question's content words appearing in the source
# chunk, the question is a lookup rather than a retrieval problem.
MAX_OVERLAP = float(os.environ.get("BENCH_MAX_OVERLAP", "0.5"))

STOP = {
    "the", "a", "an", "of", "to", "in", "for", "and", "or", "is", "are", "be",
    "this", "that", "with", "from", "it", "its", "as", "by", "on", "at", "if",
    "you", "your", "can", "will", "when", "what", "which", "how", "does", "do",
    "use", "used", "using", "return", "returns", "value", "given", "into",
}

INSTRUCTION = """Turn this documentation into one question a developer would ask.
Ask about the behaviour, not the name. Do not use the identifier.
One sentence. Return the question only."""


def content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]{4,}", text.lower()) if w not in STOP}


def harvest(db: str, want: int, rnd: random.Random) -> list[dict]:
    """Chunks carrying a substantial doc comment, with their path as truth."""
    con = sqlite3.connect(db)
    try:
        rows = con.execute(
            "SELECT path, text FROM chunks WHERE length(text) BETWEEN 400 AND 4000"
        ).fetchall()
    except sqlite3.Error:
        return []
    finally:
        con.close()

    rnd.shuffle(rows)
    out = []
    for path, text in rows:
        m = _DOC.search(text or "")
        if not m:
            continue
        doc = _TAG.sub("", m.group(1))
        doc = re.sub(r"^\s*\*\s?", "", doc, flags=re.M).strip()
        # Too short carries no meaning to paraphrase; too long is a file
        # header describing a module rather than a behaviour.
        if not (120 <= len(doc) <= 900):
            continue
        out.append({"path": path, "doc": doc, "chunk": text})
        if len(out) >= want * 6:
            break
    return out


def ask(doc: str, timeout: int = 300) -> str | None:
    body = {"model": MODEL,
            "messages": [{"role": "system", "content": INSTRUCTION},
                         {"role": "user", "content": doc[:1500]}],
            "max_tokens": 1500, "temperature": 0.3}
    try:
        req = urllib.request.Request(f"{UPSTREAM}/v1/chat/completions",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.load(r)
        msg = d["choices"][0]["message"]
        out = (msg.get("content") or "").strip()
        if not out:
            # Thinking mode can spend the budget before writing anything. The
            # question is usually the last line of the reasoning trace.
            tail = (msg.get("reasoning_content") or "").strip().splitlines()
            out = tail[-1].strip() if tail else ""
    except Exception:                                            # noqa: BLE001
        return None
    out = out.strip().strip('"').split("\n")[0].strip()
    return out or None


def keeps(q: str, item: dict) -> tuple[bool, str]:
    """Accept only a question that is answerable and not a copy of the source."""
    if not q or len(q) < 25:
        return False, "too short"
    if "?" not in q:
        return False, "not a question"
    qw = content_words(q)
    if len(qw) < 4:
        return False, "too few content words to be specific"
    overlap = len(qw & content_words(item["chunk"])) / len(qw)
    if overlap > MAX_OVERLAP:
        return False, f"leaks the source ({overlap:.0%} of words appear in it)"
    return True, ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-source", type=int, default=50)
    ap.add_argument("--seed", type=int, default=20260921)
    ap.add_argument("--out", default=os.path.join(HERE, "semantic_tasks.jsonl"))
    args = ap.parse_args()

    import repos

    sources = [(os.path.basename(e["root"]).lower(), repos.db_path(e["root"]))
               for e in repos.known()]
    pkg = os.path.join(HERE, "..", "index", "packages")
    if os.path.isdir(pkg):
        for fn in sorted(os.listdir(pkg)):
            if fn.endswith(".sqlite3") and "@" in fn:
                sources.append((fn[:-len(".sqlite3")].rpartition("@")[0].lower(),
                                os.path.join(pkg, fn)))

    rnd = random.Random(args.seed)
    tasks: list[dict] = []
    for source, db in sources:
        if not os.path.exists(db):
            continue
        pool = harvest(db, args.per_source, rnd)
        kept, rej = 0, {}
        print(f"  {source}: {len(pool)} documented chunks", flush=True)
        for item in pool:
            if kept >= args.per_source:
                break
            q = ask(item["doc"])
            if q is None:
                rej["model error"] = rej.get("model error", 0) + 1
                continue
            ok, why = keeps(q, item)
            if not ok:
                rej[why.split(" (")[0]] = rej.get(why.split(" (")[0], 0) + 1
                continue
            tasks.append({
                "id": f"semantic:{source}:{kept}",
                "kind": "semantic", "source": source,
                "query": q,
                "prompt": q,
                "truth": {"path": item["path"]},
            })
            kept += 1
            print(f"    kept {kept}/{args.per_source}", end="\r", flush=True)
        total = kept + sum(rej.values())
        print(f"    kept {kept} of {total} "
              f"({kept * 100 // max(total, 1)}% pass the leakage check)")
        for why, n in sorted(rej.items(), key=lambda x: -x[1])[:4]:
            print(f"      rejected, {why}: {n}")

    rnd.shuffle(tasks)
    with open(args.out, "w", encoding="utf-8") as f:
        for t in tasks:
            f.write(json.dumps(t) + "\n")
    print(f"\n  {len(tasks)} semantic tasks -> {os.path.abspath(args.out)}")
    if tasks:
        print(f"  example: {tasks[0]['query']}")
        print(f"     truth: {tasks[0]['truth']['path']}")


if __name__ == "__main__":
    main()
