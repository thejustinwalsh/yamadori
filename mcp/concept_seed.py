#!/usr/bin/env python
"""A random concept, drawn from the MODEL'S OWN embedding space.

After ClancyDennis/concept-seed. The mechanism is:

    os.urandom -> gaussian vector -> normalise to the unit sphere
               -> nearest whole-word token by cosine similarity

The space is the 27B's input embedding matrix, `token_embd.weight`, and the
candidates are every whole-word token in its own vocabulary -- 40,678 of
248,320 on Ternary-Bonsai-2-27B. Both are extracted once, offline, by
`scripts/extract_token_embd.py` into `index/token_embd.npz`, which refuses to
write unless known words land near related words (a wrong decode is noise and
noise has random neighbours).

WHY A RANDOM DIRECTION AND NOT A LIST. In high dimensions two random unit
vectors are almost always near-orthogonal, so consecutive draws land in
genuinely different regions of concept space. A curated list is the author's
taste in a shuffled order: the same bias every time, covering a tiny, lumpy
part of the space. The previous version of this file claimed the first and
shipped the second -- on Windows it had no dictionary and fell back to 161
hard-coded nouns, embedded with the RETRIEVAL model rather than this one.

SPREAD WITHIN ONE FAN-OUT. `draw(n)` does not take n independent directions;
it takes n exactly ORTHOGONAL ones (QR of a gaussian matrix, a Haar-random
orthonormal frame), so the n samples of one fan-out start as far apart as the
space allows, and a repeated word is redrawn.

Measured on the extracted matrix, 20,000 independent draws: 15,767 distinct
words, none drawn more than 5 times. Hubness -- a few words capturing most
nearest-neighbour queries, the usual failure of this construction in high
dimensions -- is not a problem here, and centring the matrix changed nothing.

DISTANCE FROM THE PROMPT is the point: the seed exists to pull the model
somewhere in its own space it would not otherwise go. Measured on three coding
prompts (100 fan-outs of 3 each), uniform draws land at the MEDIAN similarity
to the prompt -- the 51st percentile, cosine ~0.02 against ~0.20 for the
prompt's own nearest words -- unrelated, but not far, and one occasionally
drifts toward the prompt's neighbourhood (max 0.148). `away_from=<prompt
text>` keeps only words in the less-similar half of the vocabulary relative to
the prompt's centroid (the mean embedding of its words that are whole-word
tokens), redrawing the rest. Whether far beats merely unrelated is not yet
measured.

INJECTION POINT MATTERS. The original reports that in the system message "the
model sometimes ignored it and fell back to its default template", while in
the user message "it couldn't". `phrase()` is for the USER message.

THE NUMBER. Every word also has a u32, `encode(word)`: 32-bit FNV-1a over its
UTF-8 bytes. It is what the dashboard shows beside the word and what the
bonsai's PRNG is seeded with (design/BONSAI-VIZ.md), so the same word grows
the same limb on every machine. FNV-1a is chosen because it is six lines in
JavaScript and has no platform-dependent behaviour.

THE LAST WORD USED. `phrase()` records the word it was called with in
`index/concept_seed_last.json`, which `vitals.snapshot()` reads. Recording
happens at the injection point, because a word that was drawn but never put in
a prompt was not used.

DIFFERENCES FROM THE ORIGINAL (github.com/ClancyDennis/concept-seed, read
2026-09-22). Same pipeline; these are the departures, and why:

  space        Original: GloVe, 100-d, ~33K words (top 50K by frequency, in
               the system dictionary, no stopwords), or OpenAI
               text-embedding-3-small over 100K DBpedia concepts. Here: the
               served model's own input embeddings, 5120-d, so the direction
               is in the space the model itself reads.
  vocabulary   No dictionary filter exists on this machine, so candidates are
               whole-word BPE tokens. Token id roughly tracks merge order:
               ids below ~150K are mostly English, above are mostly other
               languages. Some candidates are word FRAGMENTS a dictionary
               would have removed ("fundra", "depos", "irrig").
  spread       Original: independent draws, no distance constraint. Here: an
               orthogonal frame per fan-out, plus `away_from`.
  template     Same as the original: "Inspiration word: X", in the user
               message, no hedge.
  per turn     Original also tests a fresh seed per turn, and draft with one
               seed then revise with another. Here: one seed per fan-out
               sample.

EVIDENCE. The source reports qualitative results and states plainly that it
has no quantitative diversity metrics. So this is a promising mechanism, not
an established result; see scripts/eval_fanout.py before it ships.
"""
from __future__ import annotations

import json
import os
import re
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HERE = os.path.dirname(os.path.abspath(__file__))
MATRIX = os.environ.get("CONCEPT_SEED_MATRIX",
                        os.path.join(HERE, "..", "index", "token_embd.npz"))
LAST = os.environ.get("CONCEPT_SEED_LAST",
                      os.path.join(HERE, "..", "index", "concept_seed_last.json"))

_lock = threading.Lock()
_record_lock = threading.Lock()
_words: list[str] | None = None
_ids: np.ndarray | None = None
_mat: np.ndarray | None = None


class NotExtracted(RuntimeError):
    """The model's embedding matrix has not been extracted on this machine."""


def _load() -> tuple[list[str], np.ndarray, np.ndarray]:
    global _words, _ids, _mat
    with _lock:
        if _mat is None:
            if not os.path.exists(MATRIX):
                raise NotExtracted(
                    f"{MATRIX} does not exist | retryable: no | remedy "
                    "(operator): run scripts/extract_token_embd.py once")
            z = np.load(MATRIX)
            _words = [str(w) for w in z["words"]]
            _ids = z["word_ids"]
            # fp16 on disk; fp32 for the product so argmax is not decided by
            # half-precision rounding.
            _mat = z["word_mat"].astype(np.float32)
        return _words, _ids, _mat


def available() -> bool:
    return _mat is not None or os.path.exists(MATRIX)


def encode(word: str) -> int:
    """32-bit FNV-1a of the UTF-8 bytes. Mirrored in the dashboard."""
    h = 0x811C9DC5
    for b in word.encode("utf-8"):
        h ^= b
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


def _centroid(text: str) -> np.ndarray | None:
    """The mean model embedding of the text's words that are whole-word
    tokens, unit length. None if none of its words are in the vocabulary."""
    words, _, mat = _load()
    index = _index()
    hit = [index[w] for w in re.findall(r"[a-z]{4,14}", (text or "").lower())
           if w in index]
    if not hit:
        return None
    c = mat[hit].mean(axis=0)
    norm = float(np.linalg.norm(c))
    return c / norm if norm else None


_word_index: dict[str, int] | None = None


def _index() -> dict[str, int]:
    global _word_index
    if _word_index is None:
        words, _, _ = _load()
        _word_index = {w: i for i, w in enumerate(words)}
    return _word_index


def draw_seeds(n: int = 1, rng: np.random.Generator | None = None,
               away_from: str | None = None) -> list[dict]:
    """n distinct concepts from n mutually orthogonal random directions.

    Each is {word, token_id, u32, hex}, plus `prompt_cos` when `away_from` is
    given. Randomness comes from os.urandom unless an rng is passed, so two
    processes started in the same second do not draw the same concepts.
    """
    words, ids, mat = _load()
    cen = _centroid(away_from) if away_from else None
    sims = mat @ cen if cen is not None else None
    ceiling = float(np.median(sims)) if sims is not None else None
    if rng is None:
        rng = np.random.default_rng(int.from_bytes(os.urandom(8), "little"))
    dim = mat.shape[1]
    out: list[dict] = []
    seen: set[int] = set()
    for _ in range(32):         # redraw on a repeat, or a word too near
        need = n - len(out)
        if need <= 0:
            break
        # QR of a gaussian matrix gives an orthonormal frame; the sign fix
        # makes it uniformly (Haar) distributed rather than biased.
        g = rng.standard_normal((dim, need))
        q, r = np.linalg.qr(g)
        q *= np.sign(np.diag(r))
        picks = np.argmax(mat @ q.astype(np.float32), axis=0)
        for k in picks.tolist():
            if k in seen or len(out) >= n:
                continue
            if ceiling is not None and sims[k] > ceiling:
                continue
            seen.add(k)
            u = encode(words[k])
            s = {"word": words[k], "token_id": int(ids[k]),
                 "u32": u, "hex": f"0x{u:08X}"}
            if sims is not None:
                s["prompt_cos"] = round(float(sims[k]), 4)
            out.append(s)
    return out


def draw(n: int = 1, rng: np.random.Generator | None = None,
         away_from: str | None = None) -> list[str]:
    """n distinct concept words. See `draw_seeds` for the numbers."""
    return [s["word"] for s in draw_seeds(n, rng, away_from)]


def record(word: str, where: str = "") -> dict:
    """Note that `word` was just put in a prompt. Never raises."""
    entry = {"word": word, "u32": encode(word),
             "hex": f"0x{encode(word):08X}", "at": time.time(),
             "where": where}
    try:
        _, ids, _ = _load()
        entry["token_id"] = int(ids[_index()[word]])
    except Exception:                                            # noqa: BLE001
        entry["token_id"] = None
    # A fan-out records one seed per sample FROM PARALLEL THREADS. With one
    # shared ".tmp" name, two writes interleaved and left `{...}}` on disk --
    # invalid JSON, so last() returned None and the dashboard showed "no seed"
    # for good. Found by the live stack suite, 2026-09-22. A lock orders the
    # writers and a per-writer temp name makes each replace atomic.
    try:
        os.makedirs(os.path.dirname(os.path.abspath(LAST)), exist_ok=True)
        tmp = f"{LAST}.{os.getpid()}.{threading.get_ident()}.tmp"
        with _record_lock:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(entry, f)
            os.replace(tmp, LAST)
    except OSError:
        pass
    return entry


def last() -> dict | None:
    """The most recently used seed, or None if none ever was."""
    try:
        with open(LAST, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def seed_for(prompt: str | None = None, n: int = 1) -> list[dict]:
    """n fresh seeds for n second-brain runs, drawn away from `prompt`, each
    {word, token_id, u32, hex}. Empty -- never an exception -- when the matrix
    is not extracted: a missing seed must not cost the request.

    Operator requirement (2026-09-23): EVERY second-brain run -- fan-out's
    candidate B, its tie-breaker C, and deep thinking -- carries one, in the
    user turn, via phrase()."""
    try:
        if not available():
            return []
        return draw_seeds(n, away_from=prompt or None)
    except Exception:                                            # noqa: BLE001
        return []


def summary(seed: dict | None) -> dict | None:
    """The part of a seed a record carries: the word and its numbers."""
    if not seed:
        return None
    return {"word": seed.get("word"), "token_id": seed.get("token_id"),
            "u32": seed.get("u32")}


def phrase(concept: str, where: str = "fanout") -> str:
    """The seed as the original states it, for the USER message. Records use.

    No hedge. An earlier version told the model not to let the seed change
    its answer, which asked it to ignore the one thing the seed is for. The
    original found the system message unreliable -- the model fell back to its
    default approach -- and the user turn not, so this goes in the user turn.
    """
    record(concept, where)
    return f"\n\nInspiration word: {concept}"


if __name__ == "__main__":
    print("  drawing 12 concepts from the model's embedding space:")
    for s in draw_seeds(12):
        print(f"    {s['word']:<16} token {s['token_id']:>6}  {s['hex']}")
