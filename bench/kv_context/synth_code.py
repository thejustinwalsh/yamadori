#!/usr/bin/env python
"""A synthetic TypeScript codebase, generated from a seed, as haystack filler.

bench/longctx/haystack.py plants K=4 needles and K near-miss decoys into
filler and grades exact recall. Its default filler is real package source
(three.js last), which is fine for recall of random names but is text every
model has read. This module generates filler that no model has read: modules
of interfaces, classes, constants and small functions over a fixed programming
vocabulary, all from `random.Random(seed)`. Plug it into the same probe:

    chunks = synth_code.chunks(seed=7, n_chars=1_800_000)
    h = haystack.build_haystack(haystack.make_item(1, 3), 400_000, chunks)

THE FILLER CARRIES DISTRACTORS OF THE NEEDLE'S OWN SHAPE. Every module has
zero-argument functions that return a 4-digit integer, the exact form of a
needle (`export function velKorthaBrin() { return 4821; }`), some with the
same doc comments. Finding "the function that returns a constant" is
therefore useless; only the exact name answers the question. The names come
from an English programming vocabulary that shares no whole word with the
needle syllables (haystack._SYL), and build_haystack still refuses any filler
that contains a needle or decoy name (bench/kv_context/test_kv_context.py
checks the first 200 items of the default seeds).

Standard library only; sends nothing.
"""
from __future__ import annotations

import hashlib
import random

WORDS = (
    "buffer vertex index range frame queue cache token layer shader sample "
    "offset stride scale bound cursor packet route stream batch chunk slot "
    "lane span grid cell mesh node edge graph heap stack pool arena block "
    "page record field entry table column row schema query filter reduce "
    "merge split parse emit flush drain fetch store load commit retry limit "
    "quota budget clock timer tick epoch phase stage step pass level depth "
    "width height length count total delta ratio weight gain noise seed hash "
    "key value label tag name path file handle socket port host client server "
    "worker task job event signal state mode flag option config param input "
    "output source target origin anchor marker window view scene camera light "
    "color texture pixel image audio voice channel track clip curve spline "
    "matrix vector angle axis plane ray box sphere bounds volume density "
    "pressure velocity force mass spring damper joint body contact collider "
    "solver integrator kernel dispatch fence barrier semaphore mutex lock "
    "guard scope context session request response header payload message "
    "envelope cipher digest codec encoder decoder reader writer builder "
    "factory adapter bridge gateway router scheduler planner tracker monitor "
    "logger metric sampler profiler tracer"
).split()
VERBS = ("get compute resolve build apply update create find make read write "
         "encode decode measure select merge split normalize clamp project "
         "sample schedule allocate release attach detach").split()
AREAS = ("core render physics audio net storage ui input math sched io gpu "
         "text cache auth").split()
DOCS = ("Returns the calibration constant for this module.",
        "Resolved constant used by the scheduler.",
        "Fixed identifier for this build step.",
        "Constant table entry; do not edit by hand.",
        "Upper bound applied by the planner.",
        "Default used when the option is absent.")
TYPES = ("number", "string", "boolean", "number[]", "Map<string, number>")


def _cap(w: str) -> str:
    return w[:1].upper() + w[1:]


class _Gen:
    def __init__(self, r: random.Random):
        self.r = r

    def camel(self, n: int = 2) -> str:
        ws = [self.r.choice(WORDS) for _ in range(n)]
        return ws[0] + "".join(_cap(w) for w in ws[1:])

    def pascal(self, n: int = 2) -> str:
        return "".join(_cap(self.r.choice(WORDS)) for _ in range(n))

    def upper(self) -> str:
        return "_".join(self.r.choice(WORDS).upper() for _ in range(2))

    def fn_name(self) -> str:
        return self.r.choice(VERBS) + "".join(_cap(self.r.choice(WORDS))
                                              for _ in range(self.r.randint(1, 2)))

    def module(self, path: str, others: list[str]) -> str:
        r = self.r
        out = []
        for _ in range(r.randint(1, 3)):
            other = r.choice(others) if others else "./base"
            names = sorted({self.pascal() for _ in range(r.randint(1, 3))})
            out.append(f'import {{ {", ".join(names)} }} from "{other}";')
        out.append("")
        for _ in range(r.randint(1, 3)):
            out.append(f"export const {self.upper()} = {r.randint(2, 99999)};")
        out.append("")
        iface = self.pascal() + "Options"
        fields = sorted({self.camel() for _ in range(r.randint(2, 5))})
        out.append(f"export interface {iface} {{")
        for f in fields:
            out.append(f"  {f}{'?' if r.random() < 0.3 else ''}: {r.choice(TYPES)};")
        out.append("}")
        out.append("")
        cls = self.pascal()
        num = self.camel()
        out.append(f"export class {cls} {{")
        out.append(f"  private {num}: number;")
        out.append(f"  constructor(opts: {iface}) {{")
        out.append(f"    this.{num} = {r.randint(1, 512)};")
        out.append("  }")
        for _ in range(r.randint(1, 4)):
            m, a, loc = self.fn_name(), self.camel(1), self.camel()
            k1, k2, k3 = r.randint(2, 97), r.randint(10, 9999), r.randint(1, 999)
            out.append(f"  {m}({a}: number): number {{")
            out.append(f"    const {loc} = this.{num} * {k1} + {a};")
            out.append(f"    if ({loc} > {k2}) {{")
            out.append(f"      return {loc} - {k3};")
            out.append("    }")
            out.append(f"    return {loc};")
            out.append("  }")
        out.append("}")
        out.append("")
        for _ in range(r.randint(1, 3)):
            f, a, b = self.fn_name(), self.camel(1), self.camel(1)
            if a == b:
                b = b + "Next"
            out.append(f"export function {f}({a}: number, {b}: number): number {{")
            out.append(f"  return {a} * {r.randint(2, 97)} + {b};")
            out.append("}")
            out.append("")
        # Distractors of the needle's exact shape.
        for _ in range(r.randint(1, 3)):
            out.append("/**")
            out.append(f" * {r.choice(DOCS)}")
            out.append(" */")
            out.append(f"export function {self.fn_name()}() {{")
            out.append(f"  return {r.randint(1000, 9999)};")
            out.append("}")
            out.append("")
        return "\n".join(out)


def chunks(seed: int = 7, n_chars: int = 1_800_000) -> list[tuple[str, str]]:
    """[(label, text)] of synthetic modules totalling at least `n_chars`.

    Deterministic in (seed, n_chars). The default covers a 262,144-token
    haystack with room to spare at ~3 characters per token."""
    r = random.Random(int.from_bytes(hashlib.sha256(f"synth|{seed}".encode())
                                     .digest()[:8], "big"))
    g = _Gen(r)
    out: list[tuple[str, str]] = []
    paths: list[str] = []
    total = 0
    seen: set[str] = set()
    while total < n_chars:
        area = r.choice(AREAS)
        path = f"src/{area}/{g.camel()}.ts"
        if path in seen:
            continue
        seen.add(path)
        rel = [f"../{p.split('/')[1]}/{p.split('/')[2][:-3]}" for p in paths[-12:]]
        text = g.module(path, rel)
        out.append((path, text))
        paths.append(path)
        total += len(text)
    return out


def fingerprint(ch: list[tuple[str, str]]) -> str:
    h = hashlib.sha256()
    for label, text in ch:
        h.update(label.encode())
        h.update(b"\0")
        h.update(text.encode())
    return h.hexdigest()[:16]


if __name__ == "__main__":
    c = chunks()
    print(f"  {len(c)} modules, {sum(len(t) for _, t in c):,} chars, "
          f"fingerprint {fingerprint(c)}")
    print(c[0][1][:1200])
