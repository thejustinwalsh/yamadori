#!/usr/bin/env python
"""Is `judge` calibrated enough to expose to a model?

A judge the model trusts and that is wrong is worse than no judge: it converts
an open question into a confident wrong answer, and nothing downstream knows.
So this has to pass before `judge` stays on the tool surface.

Two tests, because Laya failed one framing already and passing the other is
what decides whether the tool is scoped wrongly rather than broken:

  DISCRIMINATION  the same passage, a real question vs a shuffled one. If the
                  score does not move, it is reading the passage and ignoring
                  the question, which is exactly how it rated the nonsense
                  query "quantum teapot recursion" above every real one.

  DECISIONS       yes/no engineering questions with an unambiguous answer,
                  which is what the tool actually advertises. Laya is a
                  classifier for decisions with named options; this measures it
                  at that job rather than at open-ended relevance.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mcp"))
import code_search as cs  # noqa: E402

# (state, question, expected) -- expected True means "probability should be high"
DECISIONS = [
    ("function add(a: number, b: number) { return a - b; }",
     "The function body matches its name.", False),
    ("function add(a: number, b: number) { return a + b; }",
     "The function body matches its name.", True),
    ("const xs: number[] = []; for (let i = 0; i <= xs.length; i++) { use(xs[i]); }",
     "This loop reads past the end of the array.", True),
    ("const xs: number[] = []; for (let i = 0; i < xs.length; i++) { use(xs[i]); }",
     "This loop reads past the end of the array.", False),
    ("free(ptr); printf(\"%s\", ptr);",
     "This code uses memory after it has been freed.", True),
    ("printf(\"%s\", ptr); free(ptr);",
     "This code uses memory after it has been freed.", False),
    ("export function f(): string { return 42; }",
     "The return type is wrong.", True),
    ("export function f(): number { return 42; }",
     "The return type is wrong.", False),
    ("let mut v = vec![1]; let r = &v[0]; v.push(2); println!(\"{}\", r);",
     "This violates Rust's borrow rules.", True),
    ("let v = vec![1]; let r = &v[0]; println!(\"{}\", r);",
     "This violates Rust's borrow rules.", False),
]

PASSAGES = [
    "function packAtlas(glyphs: Glyph[], size: number): Atlas { /* shelf packing */ }",
    "pub unsafe extern \"C\" fn glyph_free(ptr: *mut u8, len: usize) { drop(Vec::from_raw_parts(ptr, len, len)); }",
    "export type DeepReadonly<T> = { readonly [K in keyof T]: DeepReadonly<T[K]> };",
]
REAL = [
    "This code packs glyphs into a texture atlas.",
    "This code frees memory that crossed the wasm boundary.",
    "This code defines a recursive mapped type.",
]


def noul(state: str, question: str) -> float | None:
    try:
        d = cs._post_json(cs.LAYA_URL + "/decide",
                          {"state": state,
                           "questions": {"q": {"type": "noul", "instructions": question}}},
                          timeout=20)
        return float(d["answers"]["q"]["noul"])
    except Exception as e:                                       # noqa: BLE001
        print(f"  laya unavailable: {e}")
        return None


def main() -> None:
    print("=== DISCRIMINATION: same passage, right question vs wrong question ===")
    gaps = []
    for i, passage in enumerate(PASSAGES):
        right = noul(passage, REAL[i])
        if right is None:
            return
        # Every OTHER passage's question is a wrong question for this passage.
        wrongs = [noul(passage, REAL[j]) for j in range(len(REAL)) if j != i]
        wrongs = [w for w in wrongs if w is not None]
        gap = right - max(wrongs)
        gaps.append(gap)
        print(f"  passage {i + 1}: right={right:.2f}  worst-wrong={max(wrongs):.2f}  "
              f"gap={gap:+.2f}")
    ok_disc = all(g > 0 for g in gaps)
    print(f"  -> {'PASS' if ok_disc else 'FAIL'}: "
          f"{'the question moves the score' if ok_disc else 'scores the passage, not the pair'}")

    print("\n=== DECISIONS: unambiguous yes/no engineering questions ===")
    correct = 0
    for state, q, expect in DECISIONS:
        p = noul(state, q)
        if p is None:
            return
        got = p >= 0.5
        hit = got == expect
        correct += hit
        print(f"  {'ok  ' if hit else 'MISS'}  p={p:.2f}  want={'T' if expect else 'F'}  {q[:46]}")
    n = len(DECISIONS)
    print(f"  -> {correct}/{n} correct (coin flip is {n // 2}/{n})")

    print("\n" + "=" * 62)
    verdict = ok_disc and correct >= 8
    print("VERDICT:", "keep `judge` exposed" if verdict else
          "do NOT expose `judge` as a general tool")


if __name__ == "__main__":
    main()
