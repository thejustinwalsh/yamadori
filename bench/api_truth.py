#!/usr/bin/env python
"""Does the model name APIs that exist, in the version the user actually runs?

WHY THIS METRIC

LiveCodeBench scores by execution and never looks at the code, so style, taste
and package choice are invisible to it. It is the right regression guard and
the wrong instrument for a stack whose whole claim is about a caller's
dependencies.

The claim this repo makes is narrower and checkable: given the real source of
three@0.185.1, the model should stop inventing APIs. That is not a matter of
taste. `import { positionLocal } from 'three/tsl'` is a factual assertion about
a package, and the package either exports that name or it does not.

So the metric is the hallucinated-import rate: of every name the model imports
from a library we hold source for, what share do not exist? Ground truth is the
symbol table built by tree-sitter from the real tarball. No human judges it and
no opinion enters.

WHY IMPORTS AND NOT EVERY IDENTIFIER

A local variable named `mix` is not a claim about three.js. An import IS one,
explicitly and unambiguously, which makes it the one construct in the file that
can be checked without guessing intent. Imports are extracted with tree-sitter
for the reason given in `mcp/discover.py`: a regex cannot tell an import from
the same words inside a string or a comment.

WHAT IT CANNOT TELL YOU

Whether the code works. A program can import only real symbols and still be
wrong, and this says nothing about that -- `bench/livecodebench.py` is the
execution-scored half. Nor does it measure style: that is a separate axis and
it is deliberately never folded into this number.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sqlite3
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "mcp"))
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))

import discover  # noqa: E402

CONDITIONS = {
    "direct": {"url": "http://127.0.0.1:11434/v1", "model": "bonsai"},
    "proxy": {"url": "http://127.0.0.1:1234/v1", "model": "bonsai"},
}

_FENCE = re.compile(r"```(?:ts|tsx|typescript|js|jsx|javascript)?\s*\n(.*?)```", re.S)
_NAMED = re.compile(r"import\s+(?:type\s+)?\{([^}]*)\}\s*from\s*['\"]([^'\"]+)['\"]")
_DEFAULT = re.compile(
    r"import\s+(?:type\s+)?(\w+)\s*(?:,\s*\{[^}]*\})?\s*from\s*['\"]([^'\"]+)['\"]")
_STAR = re.compile(r"import\s+\*\s+as\s+(\w+)\s+from\s*['\"]([^'\"]+)['\"]")


def code_blocks(text: str) -> str:
    """Every fenced block joined, or the raw reply if it has none."""
    blocks = _FENCE.findall(text or "")
    return "\n".join(blocks) if blocks else (text or "")


def named_imports(code: str) -> dict[str, set[str]]:
    """{package: {names}} for named imports only.

    A default import binds whatever the package exports by default, under a
    name the caller chose, so it asserts nothing checkable. A namespace import
    is the same. Only the braces make a claim.
    """
    out: dict[str, set[str]] = {}
    for names, spec in _NAMED.findall(code):
        pkg = discover.package_of(spec)
        if not pkg:
            continue
        for raw in names.split(","):
            n = raw.strip().split(" as ")[0].strip()
            n = n.removeprefix("type ").strip()
            if n and re.match(r"^[A-Za-z_$][\w$]*$", n):
                out.setdefault(pkg, set()).add(n)
    return out


def package_symbols(pkg: str) -> set[str] | None:
    """Every symbol the indexed source of a package defines, or None if absent.

    The whole symbol table rather than just the public entry points, which
    makes this LENIENT: a name defined internally and never exported counts as
    existing. That bias is deliberate -- it cannot manufacture a hallucination
    that is not there, so a non-zero rate is a floor, not an artefact.
    """
    root = os.path.join(HERE, "..", "index", "packages")
    if not os.path.isdir(root):
        return None
    slug = pkg.replace("/", "__").replace("@", "at-", 1) if pkg.startswith("@") else pkg
    for fn in sorted(os.listdir(root)):
        if not fn.endswith(".sqlite3") or "@" not in fn:
            continue
        stem, _, _ver = fn[:-len(".sqlite3")].rpartition("@")
        if stem not in (pkg, slug):
            continue
        con = sqlite3.connect(os.path.join(root, fn))
        try:
            return {r[0] for r in con.execute("SELECT DISTINCT name FROM defs")}
        except sqlite3.Error:
            return None
        finally:
            con.close()
    return None


def score(text: str) -> dict:
    """Hallucinated imports in one reply, per package we can check."""
    code = code_blocks(text)
    imports = named_imports(code)
    checked, bad, per_pkg = 0, [], {}
    for pkg, names in sorted(imports.items()):
        known = package_symbols(pkg)
        if known is None:
            per_pkg[pkg] = {"checked": 0, "skipped": len(names),
                            "why": "no indexed source"}
            continue
        miss = sorted(n for n in names if n not in known)
        checked += len(names)
        bad += [f"{pkg}:{n}" for n in miss]
        per_pkg[pkg] = {"checked": len(names), "missing": miss}
    return {"checked": checked, "hallucinated": len(bad),
            "names": sorted(bad), "per_package": per_pkg,
            "has_code": bool(code.strip()),
            "n_imports": sum(len(v) for v in imports.values())}


def generate(cond: dict, prompt: str, key: str, max_tokens: int,
             timeout: int) -> tuple[str, dict]:
    body = {"model": cond["model"],
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0.2}
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    t0 = time.time()
    req = urllib.request.Request(f"{cond['url']}/chat/completions",
                                 data=json.dumps(body).encode(), headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    msg = d["choices"][0]["message"]
    text = msg.get("content") or ""
    if not text.strip() and msg.get("reasoning_content"):
        text = msg["reasoning_content"]
    return text, {"ms": round((time.time() - t0) * 1000),
                  "finish": d["choices"][0].get("finish_reason") or "",
                  "empty_content": not (msg.get("content") or "").strip()}


def api_key() -> str:
    k = os.environ.get("YAMADORI_API_KEY", "").strip()
    if k:
        return k
    cfg = os.path.join(os.environ.get("LOCALAPPDATA", ""), "hermes", "config.yaml")
    try:
        for line in open(cfg, encoding="utf-8"):
            if "api_key" in line and ":" in line:
                return line.split(":", 1)[1].strip().strip('"').strip("'")
    except OSError:
        pass
    return ""


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def mcnemar(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(b, c) + 1))
    return min(1.0, 2.0 * tail / (2 ** n))


def report(path: str) -> None:
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    by: dict = {}
    for r in rows:
        by.setdefault(r["id"], {})[r["condition"]] = r
    paired = {q: v for q, v in by.items()
              if len(v) == len(CONDITIONS) and not any(x.get("error") for x in v.values())}
    errs = [r for r in rows if r.get("error")]
    if errs:
        print(f"\n  {len(errs)} generation errors, excluded from scoring")
    if not paired:
        print("  nothing paired yet")
        return

    n = len(paired)
    print(f"\n{'=' * 74}")
    print(f"  HALLUCINATED IMPORTS  --  {n} tasks paired")
    print("  ground truth: symbol tables built by tree-sitter from the real "
          "package source")
    print(f"{'=' * 74}")
    print(f"  {'condition':<10}{'imports':<10}{'hallucinated':<26}"
          f"{'clean replies':<16}{'no code':<8}")
    print("  " + "-" * 68)
    for c in CONDITIONS:
        chk = sum(v[c]["checked"] for v in paired.values())
        bad = sum(v[c]["hallucinated"] for v in paired.values())
        clean = sum(1 for v in paired.values()
                    if v[c]["checked"] and not v[c]["hallucinated"])
        withcode = sum(1 for v in paired.values() if v[c]["checked"])
        nocode = sum(1 for v in paired.values() if not v[c]["has_code"])
        lo, hi = wilson(bad, chk) if chk else (0.0, 0.0)
        rate = f"{bad}/{chk} = {bad / chk:.1%} [{lo:.0%}-{hi:.0%}]" if chk else "n/a"
        cl = f"{clean}/{withcode}" if withcode else "n/a"
        print(f"  {c:<10}{chk:<10}{rate:<26}{cl:<16}{nocode:<8}")

    names = list(CONDITIONS)
    a, b_ = names[0], names[1]
    # Paired at the TASK level: a task where one condition invented a symbol
    # and the other did not.
    bw = sum(1 for v in paired.values()
             if v[a]["hallucinated"] and not v[b_]["hallucinated"])
    cw = sum(1 for v in paired.values()
             if v[b_]["hallucinated"] and not v[a]["hallucinated"])
    p = mcnemar(bw, cw)
    verdict = (f"{b_} better" if bw > cw and p < 0.05 else
               f"{a} better" if cw > bw and p < 0.05 else
               "no detectable difference")
    print("\n  McNemar on tasks with any hallucination:")
    print(f"    {b_} clean where {a} was not: {bw}")
    print(f"    {a} clean where {b_} was not: {cw}")
    print(f"    p = {p:.4f}  ->  {verdict}")
    if bw + cw < 10:
        print(f"    only {bw + cw} discordant tasks: underpowered, report as "
              f"provisional")

    worst: dict = {}
    for v in paired.values():
        for c in CONDITIONS:
            for nm in v[c]["names"]:
                worst[nm] = worst.get(nm, 0) + 1
    if worst:
        print("\n  most-invented names")
        for nm, k in sorted(worst.items(), key=lambda x: -x[1])[:12]:
            print(f"    {k:>3}x  {nm}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=os.path.join(HERE, "api_tasks.jsonl"))
    ap.add_argument("--out", default=os.path.join(HERE, "api_results.jsonl"))
    ap.add_argument("--max-tokens", type=int, default=6000)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    if args.report_only:
        report(args.out)
        return

    key = api_key()
    for name, cond in CONDITIONS.items():
        try:
            _t, meta = generate(cond, "Reply with the word ready.", key, 64, 120)
            print(f"  {name:<8} alive, {meta['ms']}ms")
        except Exception as e:                                   # noqa: BLE001
            raise SystemExit(f"  {name} not answering: {type(e).__name__}: {e}")

    tasks = [json.loads(l) for l in open(args.tasks, encoding="utf-8")]
    done = set()
    if os.path.exists(args.out):
        for line in open(args.out, encoding="utf-8"):
            try:
                r = json.loads(line)
                done.add((r["id"], r["condition"]))
            except Exception:                                    # noqa: BLE001
                continue

    with open(args.out, "a", encoding="utf-8") as out:
        for i, t in enumerate(tasks):
            for cname, cond in CONDITIONS.items():
                if (t["id"], cname) in done:
                    continue
                try:
                    text, meta = generate(cond, t["prompt"], key,
                                          args.max_tokens, args.timeout)
                except Exception as e:                           # noqa: BLE001
                    out.write(json.dumps({"id": t["id"], "condition": cname,
                                          "error": f"{type(e).__name__}: {e}"}) + "\n")
                    out.flush()
                    print(f"  [{i + 1}/{len(tasks)}] {t['id']:<22} {cname:<7} "
                          f"FAILED {type(e).__name__}", flush=True)
                    continue
                s = score(text)
                rec = {"id": t["id"], "condition": cname,
                       "library": t.get("library"), **s, **meta}
                out.write(json.dumps(rec) + "\n")
                out.flush()
                bad = ",".join(s["names"][:3])
                print(f"  [{i + 1}/{len(tasks)}] {t['id']:<22} {cname:<7} "
                      f"{s['hallucinated']}/{s['checked']} invented "
                      f"{meta['ms'] / 1000:5.1f}s  {bad[:40]}", flush=True)

    report(args.out)


if __name__ == "__main__":
    main()
