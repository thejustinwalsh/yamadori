#!/usr/bin/env python
"""A random concept, drawn from the embedding space rather than from a list.

After ClancyDennis/concept-seed. The mechanism is:

    os.urandom -> gaussian vector -> normalise to the unit sphere
               -> nearest real word by cosine similarity

Why that beats picking from a curated word list: in high dimensions two random
unit vectors are almost always near-orthogonal, so consecutive draws land in
genuinely different regions of concept space. A hand-written list is the
author's taste in a shuffled order, which is the same bias every time and
covers a tiny, lumpy part of the space.

INJECTION POINT MATTERS. The original reports testing both and finding that in
the system message "the model sometimes ignored it and fell back to its default
template", while in the user message "it couldn't". The first version here put
it in the system message, which is exactly the documented failure.

EVIDENCE. The source reports qualitative results -- ten identical jokes becoming
ten different ones, three runs of a story converging versus diverging -- and
states plainly that it has no quantitative diversity metrics and no significance
testing. So this is a promising mechanism, not an established result, and it
gets measured here before it ships: see scripts/eval_fanout.py.

The vocabulary is built from the embedding model's own tokenizer output rather
than shipping GloVe, so there is no extra download and the concepts live in the
same space the retriever already uses.
"""
from __future__ import annotations

import os
import re
import sqlite3
import sys
import threading

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.environ.get("CONCEPT_SEED_DB",
                       os.path.join(HERE, "..", "index", "concepts.sqlite3"))

# Concrete, picturable nouns work better as inspiration than abstract or
# functional words: "harbour" opens a direction, "however" does not.
_WORD_OK = re.compile(r"^[a-z]{4,12}$")

_STOP = set("""about above after again against because before being below between
both during each further having here into itself more most once only other over
same some such than that their them then there these they this those through
under until very were what when where which while whom your yours ourselves
themselves everything something anything nothing someone anyone everyone""".split())

_lock = threading.Lock()
_words: list[str] | None = None
_mat: np.ndarray | None = None


def _vocab_source() -> list[str]:
    """Words to embed. Prefers a system dictionary, falls back to a builtin."""
    for p in ("/usr/share/dict/words", "/usr/dict/words"):
        try:
            with open(p, encoding="utf-8", errors="ignore") as f:
                ws = [w.strip().lower() for w in f]
            ws = [w for w in ws if _WORD_OK.match(w) and w not in _STOP]
            if len(ws) > 2000:
                return sorted(set(ws))
        except OSError:
            pass
    # Fallback: concrete nouns across unrelated domains. Smaller than ideal,
    # and the reason a real dictionary is preferred -- coverage is the whole
    # point of drawing from the space rather than from a list.
    return sorted(set("""anchor amber anvil arbor ardor ashes aurora basalt
    beacon bellows birch bison blade bloom bramble brine bronze burrow cactus
    cairn canopy canyon cedar chalk cinder cistern clover cobalt comet copper
    coral crater crystal cypress delta dune ember ember fathom fennel fern
    ferry fjord flint forge fossil fresco frost gable galley garnet geyser
    glacier granite grotto gully gypsum harbor harvest hearth heron hollow
    ingot inlet iris ivory jetty juniper kelp kiln lantern lattice lichen
    loam lotus lumber magma mangrove marble marsh meadow meridian mica mimosa
    moraine mortar moss nectar nettle nomad oasis obelisk obsidian orchard
    ospreys otter pagoda pampas papyrus pasture peat pebble pelican pewter
    pigment pillar pinion plume pollen pumice quarry quartz quill ravine reef
    resin ridge rivet rubble saffron sandbar sapling sextant shale sienna
    silt slate sluice spindle spire spruce stalk steppe stone summit tallow
    talon tannin tapestry teak thicket thistle thorn tidal timber tinder
    topaz torrent tundra turbine twine umber vellum verdigris vessel vine
    walnut warren willow windmill zephyr zircon""".split()))


def _embed(texts: list[str]) -> np.ndarray:
    import code_search as cs
    out = []
    for i in range(0, len(texts), 256):
        out.append(cs.embed(texts[i:i + 256]))
    return np.vstack(out)


def _build_cache() -> tuple[list[str], np.ndarray]:
    words = _vocab_source()
    vecs = _embed(words)
    os.makedirs(os.path.dirname(os.path.abspath(CACHE)), exist_ok=True)
    con = sqlite3.connect(CACHE)
    con.execute("CREATE TABLE IF NOT EXISTS concepts(word TEXT PRIMARY KEY, vec BLOB)")
    con.executemany("INSERT OR REPLACE INTO concepts VALUES(?,?)",
                    [(w, vecs[i].astype(np.float32).tobytes())
                     for i, w in enumerate(words)])
    con.commit()
    con.close()
    return words, vecs


def _load() -> tuple[list[str], np.ndarray]:
    global _words, _mat
    with _lock:
        if _words is not None and _mat is not None:
            return _words, _mat
        try:
            con = sqlite3.connect(CACHE)
            rows = con.execute("SELECT word, vec FROM concepts").fetchall()
            con.close()
        except sqlite3.Error:
            rows = []
        if rows:
            _words = [r[0] for r in rows]
            _mat = np.vstack([np.frombuffer(r[1], dtype=np.float32) for r in rows])
        else:
            _words, _mat = _build_cache()
        return _words, _mat


def draw(n: int = 1, rng: np.random.Generator | None = None) -> list[str]:
    """n concepts, each the nearest word to an independent random direction.

    Randomness comes from os.urandom rather than a seeded PRNG so that two
    processes started in the same second do not draw the same concept.
    """
    words, mat = _load()
    if rng is None:
        rng = np.random.default_rng(int.from_bytes(os.urandom(8), "little"))
    out: list[str] = []
    seen: set[str] = set()
    for _ in range(n * 4):
        if len(out) >= n:
            break
        v = rng.standard_normal(mat.shape[1]).astype(np.float32)
        v /= np.linalg.norm(v) or 1.0
        w = words[int(np.argmax(mat @ v))]
        if w not in seen:
            seen.add(w)
            out.append(w)
    return out


def phrase(concept: str) -> str:
    """Appended to the USER message, not the system message.

    The original found the system message unreliable: the model would ignore
    the seed and fall back to its default approach. In the user turn it cannot.
    """
    return (f"\n\n(Unrelated seed concept, to vary your approach only: "
            f"'{concept}'. Do not mention it and do not let it change the "
            f"answer -- only the route you take to it.)")


if __name__ == "__main__":
    print("  drawing 12 concepts from the embedding space:")
    for w in draw(12):
        print("   ", w)
