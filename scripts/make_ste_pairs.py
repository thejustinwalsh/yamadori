#!/usr/bin/env python
"""Generate (ordinary English -> controlled English) pairs, verifier-gated.

WHY THIS SHAPE

Rewriting into ASD-STE100 is not a rule problem. Measured on a sample of real
sentences, only 27% of violations are mechanically fixable -- contractions,
semicolons, dashes, sentence length. The other 72% need to know what the
writer meant: "should verify" could be "must verify", "can verify" or
"verifies", and only intent decides.

So rules cannot do the rewriting. What rules CAN do, exactly and for free, is
tell whether a candidate rewrite is compliant. Detection is deterministic even
where repair is not.

That asymmetry is the whole design. The local model proposes a rewrite, the
linter judges it, and only pairs that pass are kept. The judge is a regex, so
the labels are ground truth rather than opinion -- the same arrangement as
every other verified loop in this stack, where a compiler decides and nothing
is trusted because it sounds right.

WHAT IT IS FOR

A small sequence model trained on enough of these pairs does the rewrite in
one pass at classifier speed, with no extra turns and no tokens spent in the
caller's context. That is the thing worth having: English in, controlled
English out, fast. This script produces its training set.

Sentences are drawn from indexed source comments and documentation, so the
distribution matches the text this stack actually has to rewrite.
"""
from __future__ import annotations

import json
import os
import random
import re
import sqlite3
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "mcp"))
import plain  # noqa: E402

UPSTREAM = os.environ.get("LLAMA_STACK_URL", "http://127.0.0.1:11434")
MODEL = os.environ.get("AGENT_MODEL", "bonsai-agent")
OUT = os.path.join(HERE, "..", "index", "ste_pairs.jsonl")

# The instruction is itself written in controlled English. Asking for plain
# language in hedged language is the obvious own goal.
# Short on purpose. The first version listed all eight rules and the model
# spent its whole budget in reasoning_content, returning empty content on
# every one of 48 attempts. An instruction demanding brevity that is itself
# long is the obvious own goal, and this model punishes it.
INSTRUCTION = """Rewrite in simple technical English.
Active voice. Simple tenses. No contractions, semicolons or dashes.
No should, would, may, might, could. 20 words or fewer per sentence.
Keep the meaning exact. Return the rewrite only."""

_PROSE = re.compile(r"^[A-Z][^{}<>=;]{40,240}[.!?]$")


def harvest(dbs: list[str], want: int, seed: int = 0) -> list[str]:
    """Sentences from comments and documentation in the indexed corpora."""
    rnd = random.Random(seed)
    found: list[str] = []
    for db in dbs:
        if not os.path.exists(db):
            continue
        try:
            con = sqlite3.connect(db)
            rows = con.execute(
                "SELECT text FROM chunks WHERE length(text) BETWEEN 200 AND 3000"
            ).fetchall()
            con.close()
        except sqlite3.Error:
            continue
        rnd.shuffle(rows)
        for (text,) in rows:
            for raw in re.split(r"(?<=[.!?])\s+", text):
                line = re.sub(r"^[\s*/#-]+", "", raw).strip()
                if not _PROSE.match(line):
                    continue
                # Only sentences that actually violate something are useful;
                # a compliant sentence teaches the identity function.
                if plain.check(line):
                    found.append(line)
            if len(found) >= want * 4:
                break
    rnd.shuffle(found)
    return found[:want * 4]


def rewrite(sentence: str, timeout: int = 300) -> str | None:
    body = {"model": MODEL,
            "messages": [{"role": "system", "content": INSTRUCTION},
                         {"role": "user", "content": sentence}],
            "max_tokens": 1200, "temperature": 0.2}
    try:
        req = urllib.request.Request(f"{UPSTREAM}/v1/chat/completions",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.load(r)
        msg = d["choices"][0]["message"]
        out = (msg.get("content") or "").strip()
        if not out:
            # Thinking mode can consume the budget and return empty content.
            # The rewrite is often the last line of the reasoning trace.
            tail = (msg.get("reasoning_content") or "").strip().splitlines()
            out = tail[-1].strip() if tail else ""
    except Exception:                                            # noqa: BLE001
        return None
    out = out.strip('"').strip()
    return out or None


def keeps(original: str, candidate: str) -> tuple[bool, str]:
    """Accept only a rewrite that is compliant AND still says the thing.

    Compliance alone is not enough: "It works." is perfectly compliant and has
    thrown away the content. Requiring shared content words is a coarse guard
    against a rewrite that is clean and wrong.
    """
    if not candidate:
        return False, "empty"
    if plain.check(candidate):
        return False, "still violates"
    words = lambda s: {w.lower().strip(".,:;()") for w in s.split() if len(w) > 4}
    a, b = words(original), words(candidate)
    if a and len(a & b) / len(a) < 0.4:
        return False, "lost the content"
    if len(candidate) < len(original) * 0.4:
        return False, "too short"
    return True, ""


def main() -> None:
    import repos
    dbs = [repos.db_path(e["root"]) for e in repos.known()]
    pkg = os.path.join(HERE, "..", "index", "packages")
    if os.path.isdir(pkg):
        dbs += [os.path.join(pkg, f) for f in os.listdir(pkg) if f.endswith(".sqlite3")]

    want = int(os.environ.get("STE_PAIRS", "40"))
    candidates = harvest(dbs, want)
    print(f"  {len(candidates)} non-compliant sentences harvested")

    kept, rejected = [], {}
    for s in candidates:
        if len(kept) >= want:
            break
        out = rewrite(s)
        if out is None:
            rejected["model error"] = rejected.get("model error", 0) + 1
            continue
        ok, why = keeps(s, out)
        if ok:
            kept.append({"ordinary": s, "ste": out,
                         "violations_before": len(plain.check(s))})
            print(f"  kept {len(kept)}/{want}", end="\r", flush=True)
        else:
            rejected[why] = rejected.get(why, 0) + 1

    with open(OUT, "w", encoding="utf-8") as f:
        for row in kept:
            f.write(json.dumps(row) + "\n")
    total = len(kept) + sum(rejected.values())
    print(f"\n  kept {len(kept)} of {total} attempts "
          f"({len(kept) * 100 // max(total, 1)}% pass the verifier)")
    for why, n in sorted(rejected.items(), key=lambda x: -x[1]):
        print(f"    rejected, {why}: {n}")
    print(f"  written {os.path.abspath(OUT)}")


if __name__ == "__main__":
    main()
