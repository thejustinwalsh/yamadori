#!/usr/bin/env python
"""Does answering four ways beat answering once?

The claim being tested is that consensus across diverse variants finds the
right file more often than a single answer does, at 2.7x the wall clock. If it
does not, the fan-out is pure latency and should not ship.

Graded on file paths, which are checkable, not on prose quality. Each question
has a known correct file in the gauntlet repositories.
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mcp"))
import fanout  # noqa: E402
import proxy  # noqa: E402
import repos  # noqa: E402

KOOTA = "C:/Users/jwals/gauntlet/koota"

# (question, substring the correct path must contain)
TASKS = [
    ("Where is the Trait type defined?", "trait"),
    ("Where is the World created?", "world"),
    ("Where are entities allocated or spawned?", "entit"),
    ("Where is a query matched against entities?", "quer"),
    ("Where is the relation helper defined?", "relation"),
    ("Where does a trait get added to an entity?", "trait"),
    ("Where is the entity index or sparse set kept?", "entit"),
    ("Where are systems or schedules run?", "system"),
]


def ask(question: str, n: int) -> tuple[str, float]:
    body = {
        "model": "bonsai-agent",
        "messages": [
            {"role": "system",
             "content": f"You are a coding agent.\nWorking directory: {KOOTA}"},
            {"role": "user", "content": question + " Answer with the file path."},
        ],
        "max_tokens": 700,
    }
    t0 = time.time()
    if n == 1:
        d = proxy.complete(body)
        content = d["choices"][0]["message"].get("content") or ""
    else:
        # Let the proxy resolve tools and repo first, then fan the resolved
        # request out, so both arms get identical retrieval.
        prepared = proxy.prepare(body)
        v = fanout.run(prepared, n=n)
        content = (v.get("winner") or {}).get("content", "")
    return content, time.time() - t0


def hit(text: str, want: str) -> bool:
    return want.lower() in (text or "").lower()


def main() -> None:
    repos.ensure(KOOTA, background=False, from_trusted=True)
    print(f"  {len(TASKS)} questions against koota\n")
    rows = []
    for q, want in TASKS:
        a1, t1 = ask(q, 1)
        a4, t4 = ask(q, 4)
        h1, h4 = hit(a1, want), hit(a4, want)
        rows.append((q, h1, h4, t1, t4))
        print(f"  N=1 {'ok ' if h1 else 'MISS'}  N=4 {'ok ' if h4 else 'MISS'}  "
              f"{t1:5.1f}s / {t4:5.1f}s   {q[:44]}")

    s1 = sum(r[1] for r in rows)
    s4 = sum(r[2] for r in rows)
    w1 = sum(r[3] for r in rows)
    w4 = sum(r[4] for r in rows)
    n = len(rows)
    print("\n" + "=" * 62)
    print(f"  single answer   : {s1}/{n}   {w1:.0f}s total")
    print(f"  four-way consensus: {s4}/{n}   {w4:.0f}s total  ({w4 / max(w1, 1):.1f}x slower)")
    only4 = sum(1 for r in rows if r[2] and not r[1])
    only1 = sum(1 for r in rows if r[1] and not r[2])
    print(f"  discordant: consensus-only {only4}, single-only {only1}")
    if only4 > only1:
        print("  -> fan-out wins the discordant pairs")
    elif only1 > only4:
        print("  -> fan-out LOSES; it is pure latency, do not ship it")
    else:
        print("  -> indistinguishable at this n; not worth the latency yet")


if __name__ == "__main__":
    main()
