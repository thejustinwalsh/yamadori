#!/usr/bin/env python
"""concept_seed, asserted against a synthetic matrix. No GPU, no GGUF.

The real matrix is 400 MB and extracted per machine; its decode is gated by
scripts/extract_token_embd.py's neighbour probes. What is asserted here is
everything built on top of it: the number, the spread, the distance from the
prompt, and the record the dashboard reads.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

_TMP = tempfile.mkdtemp(prefix="yamadori_test_seed_")
os.environ["CONCEPT_SEED_MATRIX"] = os.path.join(_TMP, "token_embd.npz")
os.environ["CONCEPT_SEED_LAST"] = os.path.join(_TMP, "last.json")

# 400 words in 64 dimensions: enough for orthogonal frames and a median.
_rng = np.random.default_rng(0)
WORDS = [f"word{i:03d}" for i in range(396)] + ["shader", "water",
                                                "caustics", "render"]
_mat = _rng.standard_normal((len(WORDS), 64)).astype(np.float32)
_mat /= np.linalg.norm(_mat, axis=1, keepdims=True)
np.savez(os.environ["CONCEPT_SEED_MATRIX"], words=np.array(WORDS),
         word_ids=np.arange(1000, 1000 + len(WORDS), dtype=np.int32),
         word_mat=_mat.astype(np.float16))

import concept_seed  # noqa: E402
import vitals  # noqa: E402

_results: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> bool:
    _results.append((bool(ok), name, detail))
    return bool(ok)


def test_the_number_is_fnv1a_32():
    # Published FNV-1a 32-bit test vectors.
    for word, want in (("", 0x811C9DC5), ("a", 0xE40C292C),
                       ("foobar", 0xBF9CF968)):
        got = concept_seed.encode(word)
        check(got == want, f"encode({word!r}) is the FNV-1a vector",
              f"0x{got:08X} != 0x{want:08X}")
    check(concept_seed.encode("água") == concept_seed.encode("água"),
          "and it is a pure function of the UTF-8 bytes")


def test_a_fanout_is_distinct_and_spread():
    seeds = concept_seed.draw_seeds(5)
    words = [s["word"] for s in seeds]
    check(len(words) == 5 and len(set(words)) == 5,
          "draw_seeds(5) returns five distinct words", str(words))
    s = seeds[0]
    check(s["hex"] == f"0x{s['u32']:08X}"
          and s["u32"] == concept_seed.encode(s["word"])
          and s["token_id"] == 1000 + WORDS.index(s["word"]),
          "each carries its u32, hex and model token id", json.dumps(s))
    a = concept_seed.draw(4, np.random.default_rng(42))
    b = concept_seed.draw(4, np.random.default_rng(42))
    check(a == b, "a seeded rng reproduces the draw exactly", f"{a} {b}")
    # Orthogonal directions: over many fan-outs, words inside one fan-out
    # should be no more similar than chance.
    rng = np.random.default_rng(1)
    idx = {w: i for i, w in enumerate(WORDS)}
    cos = []
    for _ in range(200):
        ws = concept_seed.draw(3, rng)
        v = _mat[[idx[w] for w in ws]]
        g = v @ v.T
        cos += [g[0, 1], g[0, 2], g[1, 2]]
    check(abs(float(np.mean(cos))) < 0.05,
          "words within a fan-out are unrelated on average",
          f"mean cos {np.mean(cos):.3f}")


def test_away_from_keeps_the_far_half():
    prompt = "render a water caustics shader"
    idx = {w: i for i, w in enumerate(WORDS)}
    cen = _mat[[idx[w] for w in ("render", "water", "caustics", "shader")]]
    cen = cen.mean(axis=0)
    cen /= np.linalg.norm(cen)
    median = float(np.median(_mat @ cen))
    rng = np.random.default_rng(5)
    got = [s for _ in range(50)
           for s in concept_seed.draw_seeds(3, rng, away_from=prompt)]
    worst = max(s["prompt_cos"] for s in got)
    check(len(got) == 150, "away_from still fills every fan-out", str(len(got)))
    check(worst <= median + 1e-3,
          "and no word is nearer the prompt than the vocabulary median",
          f"worst {worst:.3f} median {median:.3f}")
    names = {s["word"] for s in got}
    check(not names & {"render", "water", "caustics", "shader"},
          "and the prompt's own words are never drawn")
    check(concept_seed.draw(2, away_from="zzzz qqqq"),
          "a prompt with no vocabulary words falls back to plain draws")


def test_the_last_used_word_reaches_vitals():
    check(vitals.seed() is None or isinstance(vitals.seed(), dict),
          "vitals.seed() answers before anything was recorded")
    text = concept_seed.phrase("word007", where="test")
    check(text == "\n\nInspiration word: word007",
          "phrase() is the original's plain form, with no hedge", repr(text))
    last = concept_seed.last()
    check(last and last["word"] == "word007"
          and last["u32"] == concept_seed.encode("word007")
          and last["token_id"] == 1007 and last["where"] == "test",
          "phrase() records the word it injected, with its number",
          json.dumps(last))
    check(vitals.seed() == last,
          "and vitals.snapshot()'s seed field reads the same record",
          json.dumps(vitals.seed()))


def test_parallel_fanout_records_never_corrupt_the_file():
    """A fan-out records from parallel threads. One shared temp name left
    `{...}}` on disk in the live run, so last() returned None."""
    import threading
    # Varied LENGTHS matter: the live corruption was a shorter record written
    # over a longer one through a shared file, leaving the longer one's tail.
    # Equal-length words cannot show it (a first version of this test used
    # word000..word039 and passed against the broken code).
    words = ["w" * k for k in range(1, 41)]
    # A race is probabilistic: one round caught the broken code 1 time in 5.
    # Checking after each of 25 rounds makes a miss very unlikely.
    bad = None
    for _round in range(25):
        threads = [threading.Thread(target=concept_seed.record,
                                    args=(w, "fanout")) for w in words]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        with open(concept_seed.LAST, encoding="utf-8") as f:
            raw = f.read()
        try:
            parsed = json.loads(raw)
        except ValueError:
            parsed, bad = None, raw
            break
    check(bad is None and parsed is not None and parsed.get("word") in words,
          "25 rounds of 40 parallel records always leave one valid JSON "
          "record", (bad or "")[:160])
    check(concept_seed.last() == parsed, "and last() reads it")
    leftovers = [f for f in os.listdir(os.path.dirname(concept_seed.LAST))
                 if f.endswith(".tmp")]
    check(not leftovers, "and no temp files are left behind", str(leftovers))


def main() -> int:
    for fn in (test_the_number_is_fnv1a_32,
               test_a_fanout_is_distinct_and_spread,
               test_away_from_keeps_the_far_half,
               test_the_last_used_word_reaches_vitals,
               test_parallel_fanout_records_never_corrupt_the_file):
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
    print(f"\n{'=' * 70}\n  {passed}/{len(_results)} checks passed")
    return 0 if passed == len(_results) else 1


if __name__ == "__main__":
    sys.exit(main())
