#!/usr/bin/env python
"""Build the text corpus llama-kv-mean-center calibrates the K bias on.

    python bench/kv_context/make_calib_corpus.py --out <path> [--kib 450]

Offline, standard library, deterministic: sorted files from this repo, each
cut to at most 6 KB so no single file dominates, in three equal shares --

    prose       docs/*.md
    Python      mcp/*.py (not the tests)
    TypeScript  web/src/**/*.ts(x) and index/packages/_src/typegpu@*/ (.ts)

-- which is the mix the stack serves (TS, TSL/TypeGPU, Python tooling, prose).
Prints the byte count and sha256, which docs/CONTEXT-EXPANSION.md asks to be
recorded next to the bias file.

Why a repo corpus rather than the fork's make-calib-corpus.sh: that script
starts its own llama-server (a second GPU consumer) and needs bash/curl/jq.
Its README reports that the per-channel K mean is dominated by the model's
own channel structure, not the corpus (self-generated vs a standard
multi-domain set: cosine > 0.95 on every layer, on a different model); that
is a claim about their model, and the FIT/ACCURACY phases are what test the
result on ours. The kv-mean-center tool needs >= 512 tokens (its -c); a
~450 KiB corpus is ~130k tokens, ~250 chunks.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
PER_FILE = 6_000


def _files(patterns: list[str], exclude: tuple[str, ...] = ()) -> list[str]:
    out = set()
    for p in patterns:
        out.update(glob.glob(os.path.join(ROOT, p), recursive=True))
    return sorted(f for f in out if os.path.isfile(f)
                  and not any(x in f.replace("\\", "/") for x in exclude))


def _take(files: list[str], budget: int) -> str:
    parts, n = [], 0
    for f in files:
        try:
            t = open(f, encoding="utf-8", errors="replace").read().replace("\r\n", "\n")
        except OSError:
            continue
        t = t[:PER_FILE]
        cut = t.rfind("\n")
        t = t[:cut] if cut > 0 else t
        if len(t) < 200:
            continue
        parts.append(t)
        n += len(t)
        if n >= budget:
            break
    return "\n\n".join(parts)


def build(kib: int = 450) -> str:
    share = kib * 1024 // 3
    prose = _take(_files(["docs/*.md"]), share)
    py = _take(_files(["mcp/*.py"], exclude=("/test_",)), share)
    ts = _take(_files(["web/src/**/*.ts", "web/src/**/*.tsx",
                       "index/packages/_src/typegpu@*/**/*.ts"],
                      exclude=(".d.ts", "/dist/", "__fixtures__")), share)
    return "\n\n".join([prose, py, ts]) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--kib", type=int, default=450)
    a = ap.parse_args()
    text = build(a.kib)
    data = text.encode("utf-8")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "wb") as f:
        f.write(data)
    print(f"  wrote {a.out}: {len(data):,} bytes, sha256 {hashlib.sha256(data).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
