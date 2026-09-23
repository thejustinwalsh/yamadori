#!/usr/bin/env python
"""Haystacks, needles, questions and the grader for bench/longctx.

Standard library only; nothing here talks to a server, so every property the
benchmark rests on is testable offline (bench/longctx/test_longctx.py).

THE PROBE

A haystack is a concatenation of REAL source files -- the package sources the
repo already indexes, index/packages/_src/<pkg>@<ver>/ -- in an order drawn
from the item's seed. Into it go K=4 NEEDLES and K DECOYS:

    needle   export function velKorthaBrin() {      // at a controlled depth
               return 4821;
             }
    decoy    export function velKorthaDask() { return 7310; }   // near-miss
                                                                 // name, other
                                                                 // value

Each needle sits at one of five depth bins (10/30/50/70/90% of the haystack);
each decoy shares all but the last syllable of one needle's name and sits at a
random depth. A reader that matches "roughly that name" gets the decoy's
value. Three questions per haystack:

    single   what does `A` return?                    exact integer
    multi    what does each of `A`,`B`,`C`,`D` return? all four exact
    reason   x = A's value, y = B's value; x+y (or |x-y|)?   exact integer

PAIRING ACROSS LENGTHS. Everything that defines an item -- names, values,
decoys, depth bins, which needle the single question asks about, which pair
the reason question combines -- depends only on (seed, item). Only the filler
length changes with L. So item i at 8k and item i at 128k are the same
question at a different length, and a drop is tested within item (exact
McNemar, bench/longctx/analyse.py), not between two unrelated samples.

DEPTH is measured in characters of the haystack before the needle, as a
proportion of the haystack's characters: a tokenizer proxy. Code's
chars-per-token is roughly uniform along one haystack, and the test suite
checks the proportion against a crude token count as well.
"""
from __future__ import annotations

import hashlib
import os
import random
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SRC_ROOT = os.path.join(ROOT, "index", "packages", "_src")

# The newest indexed version of each package, so near-identical copies of one
# library do not pad the haystack. three.js is last: it is contaminated for
# every model, which does not matter for filler (the needles are random
# names), but it is only needed past ~1 MB of the others.
DEFAULT_SOURCES = [
    "typegpu@0.12.5", "three-mesh-bvh@0.9.15", "koota@0.6.6",
    "wgpu-matrix@3.4.2", "postprocessing@6.39.5", "react-three__drei@10.7.8",
    "react-three__fiber@9.8.0", "typegpu__noise@0.12.0",
    "typegpu__three@0.12.1", "react-three__postprocessing@3.1.2",
    "three@0.186.0",
]
CODE_EXT = (".js", ".mjs", ".ts", ".tsx", ".jsx")
SKIP_PARTS = (".d.ts", ".min.", ".map", ".cjs")
CHUNK_CHARS = 40_000          # a larger file is split at line boundaries
MIN_FILE_CHARS = 400
MAX_LINE_CHARS = 1_000        # a longer line means minified; skipped

K = 4
DEPTH_BINS = (0.1, 0.3, 0.5, 0.7, 0.9)
TASKS = ("single", "multi", "reason")

_SYL = ("vel", "kor", "tha", "brin", "mog", "sar", "quen", "dil", "rav", "tor",
        "nex", "pul", "gam", "zor", "lith", "fen", "vok", "sim", "dar", "hol",
        "wex", "yun", "cal", "bro", "tis", "jer", "mun", "pax", "rel", "gno",
        "fyr", "lod", "quo", "sep", "trin", "ulm", "vas", "yel", "zin", "kep")


# ------------------------------------------------------------- sources -----
def _files_under(root: str) -> list[str]:
    out = []
    for d, _dirs, fs in os.walk(root):
        for f in fs:
            if not f.endswith(CODE_EXT) or any(s in f for s in SKIP_PARTS):
                continue
            out.append(os.path.join(d, f))
    return sorted(out)


def load_chunks(sources: list[str] | None = None,
                src_root: str = SRC_ROOT) -> list[tuple[str, str]]:
    """[(label, text)] -- every usable file (or 40k-char part of one), sorted.

    Deterministic: the order here depends only on the files on disk, and every
    per-item shuffle starts from it. Exact duplicate chunks are dropped."""
    sources = sources or DEFAULT_SOURCES
    seen: set[str] = set()
    chunks: list[tuple[str, str]] = []
    for pkg in sources:
        base = os.path.join(src_root, pkg)
        if not os.path.isdir(base):
            continue
        for path in _files_under(base):
            try:
                with open(path, encoding="utf-8", errors="replace") as f:
                    text = f.read().replace("\r\n", "\n")
            except OSError:
                continue
            lines = text.split("\n")
            if len(text) < MIN_FILE_CHARS or max(map(len, lines)) > MAX_LINE_CHARS:
                continue
            rel = os.path.relpath(path, src_root).replace(os.sep, "/")
            parts, cur, n = [], [], 0
            for ln in lines:
                cur.append(ln)
                n += len(ln) + 1
                if n >= CHUNK_CHARS:
                    parts.append("\n".join(cur))
                    cur, n = [], 0
            if cur:
                parts.append("\n".join(cur))
            for i, p in enumerate(parts):
                h = hashlib.sha1(p.encode("utf-8")).hexdigest()
                if h in seen or len(p.strip()) < 50:
                    continue
                seen.add(h)
                label = rel if len(parts) == 1 else f"{rel} (part {i + 1}/{len(parts)})"
                chunks.append((label, p))
    return chunks


def corpus_chars(chunks: list[tuple[str, str]]) -> int:
    return sum(len(t) for _, t in chunks)


# ------------------------------------------------------------- items -------
def _rng(seed: int, item: int, salt: str) -> random.Random:
    h = hashlib.sha256(f"{seed}|{item}|{salt}".encode()).digest()
    return random.Random(int.from_bytes(h[:8], "big"))


def _name(r: random.Random, n: int = 3) -> list[str]:
    return [r.choice(_SYL) for _ in range(n)]


def _camel(parts: list[str]) -> str:
    return parts[0] + "".join(p.capitalize() for p in parts[1:])


def make_item(seed: int, item: int) -> dict:
    """Everything that defines item `item`, independent of length.

    The single question asks about the needle in bin (item mod 5), so across
    n items each depth bin is asked about n/5 times at every length."""
    r = _rng(seed, item, "item")
    names: set[str] = set()
    needles, decoys = [], []
    values: set[int] = set()

    def fresh_value() -> int:
        while True:
            v = r.randint(1000, 9999)
            if v not in values:
                values.add(v)
                return v

    target_bin = DEPTH_BINS[item % len(DEPTH_BINS)]
    others = [b for b in DEPTH_BINS if b != target_bin]
    bins = [target_bin] + r.sample(others, K - 1)
    r.shuffle(bins)
    for b in bins:
        while True:
            parts = _name(r, 3)
            nm = _camel(parts)
            # No name may be a prefix or substring of another: the grader
            # matches names exactly, and the question must not be answerable
            # by a partial match.
            if nm not in names and all(nm not in o and o not in nm for o in names):
                break
        names.add(nm)
        while True:
            dparts = parts[:2] + [r.choice([s for s in _SYL if s != parts[2]])]
            dn = _camel(dparts)
            if dn not in names and all(dn not in o and o not in dn for o in names):
                break
        names.add(dn)
        needles.append({"name": nm, "value": fresh_value(), "depth": b})
        decoys.append({"name": dn, "value": fresh_value(),
                       "depth": round(r.uniform(0.05, 0.95), 4), "of": nm})
    single = next(n for n in needles if n["depth"] == target_bin)
    a, b = r.sample(needles, 2)
    op = r.choice(["sum", "absdiff"])
    ans = a["value"] + b["value"] if op == "sum" else abs(a["value"] - b["value"])
    order = [n["name"] for n in needles]
    r.shuffle(order)
    return {"seed": seed, "item": item, "needles": needles, "decoys": decoys,
            "single": {"name": single["name"], "value": single["value"],
                       "depth": single["depth"]},
            "multi": {"order": order},
            "reason": {"a": a["name"], "b": b["name"], "op": op, "value": ans,
                       "depths": [a["depth"], b["depth"]]}}


_COMMENTS = ("Returns the calibration constant for this module.",
             "Resolved constant used by the scheduler.",
             "Fixed identifier for this build step.",
             "Constant table entry; do not edit by hand.")


def needle_code(name: str, value: int, r: random.Random) -> str:
    c = r.choice(_COMMENTS)
    return (f"\n/**\n * {c}\n */\nexport function {name}() {{\n"
            f"  return {value};\n}}\n\n")


# ------------------------------------------------------------- haystack ----
def filler(seed: int, item: int, n_chars: int,
           chunks: list[tuple[str, str]]) -> str:
    """`n_chars` of real source, in the item's order, cut at a line boundary.

    The order depends on (seed, item) only, so a longer haystack for the same
    item starts with the same files as a shorter one. Wraps the corpus if it
    is exhausted (each pass reshuffled), which a 256k haystack of the default
    sources does not need."""
    if not chunks:
        raise ValueError("no source chunks: is index/packages/_src present?")
    out, n, rnd = [], 0, 0
    while n < n_chars:
        order = list(range(len(chunks)))
        _rng(seed, item, f"order{rnd}").shuffle(order)
        for i in order:
            label, text = chunks[i]
            piece = f"// ==== file: {label} ====\n{text}\n\n"
            if n + len(piece) > n_chars:
                room = n_chars - n
                cut = piece.rfind("\n", 0, room)
                piece = piece[:cut + 1] if cut > 0 else ""
                out.append(piece)
                n += len(piece)
                break
            out.append(piece)
            n += len(piece)
        rnd += 1
        if rnd > 50:
            break
    return "".join(out)


def _boundary(text: str, pos: int, window: int) -> int:
    """The insertion point nearest `pos`: a blank line if one is within
    `window` characters, else the nearest line start. Never mid-line."""
    pos = max(0, min(len(text), pos))
    lo, hi = max(0, pos - window), min(len(text), pos + window)
    best = None
    for m in re.finditer(r"\n\n", text[lo:hi]):
        p = lo + m.start() + 1
        if best is None or abs(p - pos) < abs(best - pos):
            best = p
    if best is not None:
        return best
    left = text.rfind("\n", 0, pos)
    right = text.find("\n", pos)
    cands = [p + 1 for p in (left, right) if p >= 0]
    if not cands:
        return 0
    return min(cands, key=lambda p: abs(p - pos))


def build_haystack(item: dict, n_chars: int,
                   chunks: list[tuple[str, str]]) -> dict:
    """The haystack text for one item at one size, and where each inserted
    function actually landed (as a proportion of the final text)."""
    r = _rng(item["seed"], item["item"], "code")
    inserts = ([("needle", n) for n in item["needles"]]
               + [("decoy", d) for d in item["decoys"]])
    codes = {x["name"]: needle_code(x["name"], x["value"], r) for _, x in inserts}
    ins_chars = sum(len(c) for c in codes.values())
    base = filler(item["seed"], item["item"], max(n_chars - ins_chars, 1000), chunks)
    for _, x in inserts:
        if x["name"] in base:
            raise ValueError(f"needle name {x['name']} occurs in the filler")
    # Place against the FINAL length: the target offset of an insert is its
    # depth times the final size, minus the inserts that precede it.
    final_len = len(base) + ins_chars
    window = max(200, final_len // 200)            # 0.5% of the haystack
    plan = sorted(inserts, key=lambda kx: kx[1]["depth"])
    pieces, prev, placed_before = [], 0, 0
    marks = []
    for kind, x in plan:
        want = int(x["depth"] * final_len) - placed_before
        at = _boundary(base, max(want, prev), window)
        at = max(at, prev)
        pieces.append(base[prev:at])
        marks.append((kind, x))
        pieces.append(codes[x["name"]])
        placed_before += len(codes[x["name"]])
        prev = at
    pieces.append(base[prev:])
    text = "".join(pieces)
    landed = {}
    for kind, x in marks:
        at = text.index(f"export function {x['name']}()")
        landed[x["name"]] = {"kind": kind, "target": x["depth"],
                             "actual": round(at / len(text), 4)}
    return {"text": text, "landed": landed, "chars": len(text)}


# ------------------------------------------------------------- prompts -----
# The nonce comes FIRST, so a fresh nonce shares no prompt prefix with any
# earlier request beyond the chat template's own opening tokens.
SYSTEM = ("Session {nonce}. You are a careful code reader. Answer only from "
          "the code you are given.")
ANSWER_RULE = ("End your reply with exactly one line of the form\n"
               "ANSWER: {shape}")


def question(item: dict, task: str) -> str:
    if task == "single":
        s = item["single"]
        return (f"What integer does the function `{s['name']}` defined in the "
                f"code above return?\n\n"
                + ANSWER_RULE.format(shape="<integer>"))
    if task == "multi":
        names = item["multi"]["order"]
        listed = ", ".join(f"`{n}`" for n in names)
        shape = ", ".join(f"{n}=<integer>" for n in names)
        return (f"Four functions are defined in the code above: {listed}. For "
                f"each one, give the integer it returns.\n\n"
                + ANSWER_RULE.format(shape=shape))
    if task == "reason":
        q = item["reason"]
        what = ("x + y" if q["op"] == "sum"
                else "the absolute difference |x - y|")
        return (f"Let x be the integer returned by the function `{q['a']}` and "
                f"y the integer returned by the function `{q['b']}`, both "
                f"defined in the code above. What is {what}?\n\n"
                + ANSWER_RULE.format(shape="<integer>"))
    raise ValueError(task)


def user_message(haystack: str, q: str) -> str:
    return ("Below is a concatenation of source files.\n\n<code>\n"
            + haystack + "\n</code>\n\n" + q)


def messages(haystack: str, q: str, nonce: str) -> list[dict]:
    return [{"role": "system", "content": SYSTEM.format(nonce=nonce)},
            {"role": "user", "content": user_message(haystack, q)}]


def expected(item: dict, task: str):
    if task == "single":
        return item["single"]["value"]
    if task == "reason":
        return item["reason"]["value"]
    if task == "multi":
        return {n["name"]: n["value"] for n in item["needles"]}
    raise ValueError(task)


# ------------------------------------------------------------- grading -----
_ANS = re.compile(r"ANSWER\s*[:=]\s*(.*)", re.I)
_INT = re.compile(r"-?\d[\d,_]*")


def _to_int(s: str) -> int | None:
    s = s.replace(",", "").replace("_", "")
    try:
        return int(s)
    except ValueError:
        return None


def _answer_tail(content: str) -> tuple[str | None, str]:
    """The text after the LAST 'ANSWER:' marker, and how it was found."""
    ms = list(_ANS.finditer(content or ""))
    if ms:
        return content[ms[-1].start(1):], "answer_line"
    return None, "none"


def grade(item: dict, task: str, content: str | None,
          finish_reason: str | None) -> dict:
    """Exact-match grading of one reply. Never raises.

    outcome: correct | wrong | budget.  A `length` finish with no ANSWER line
    is a BUDGET event, never an answer (AGENTS.md), and is not scored.
    parse: answer_line (the instructed form) | fallback (no ANSWER line; the
    last integer / name=value pairs anywhere in the reply) | none."""
    content = content or ""
    tail, how = _answer_tail(content)
    if tail is None and finish_reason == "length":
        return {"outcome": "budget", "correct": None, "parse": "none",
                "got": None}
    exp = expected(item, task)
    if task in ("single", "reason"):
        # The instructed form is one integer on the ANSWER line. A reason
        # answer written as "ANSWER: 4821 + 1234 = 6055" ends in its result,
        # so the reason task reads the line's LAST integer, single its first.
        src = tail.split("\n", 1)[0] if tail is not None else content
        nums = [_to_int(m.group(0)) for m in _INT.finditer(src)]
        nums = [x for x in nums if x is not None]
        if not nums:
            return {"outcome": "wrong", "correct": False,
                    "parse": how if tail is not None else "none", "got": None}
        if tail is None:
            got = nums[-1]
        else:
            got = nums[-1] if task == "reason" else nums[0]
        ok = got == exp
        return {"outcome": "correct" if ok else "wrong", "correct": ok,
                "parse": how if tail is not None else "fallback", "got": got}
    # multi: name=value pairs, exact names only.
    src = tail if tail is not None else content
    got: dict[str, int] = {}
    for nm in exp:
        m = re.search(r"\b" + re.escape(nm) + r"\b[`*]*(?:\(\))?[`*]*\s*"
                      r"(?:=|:|->|=>|\u2192|\breturns\b|\bis\b)\s*[`*]*(-?\d[\d,_]*)",
                      src)
        if m:
            v = _to_int(m.group(1))
            if v is not None:
                got[nm] = v
    per = {nm: got.get(nm) == v for nm, v in exp.items()}
    ok = all(per.values())
    parse = how if tail is not None else ("fallback" if got else "none")
    return {"outcome": "correct" if ok else "wrong", "correct": ok,
            "parse": parse, "got": got, "per_needle": per}
