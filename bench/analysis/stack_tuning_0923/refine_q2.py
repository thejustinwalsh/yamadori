"""Q2 refinement: model-attributable lint findings and a real compile check.

code_quality.py counted every ruff finding. Some are not the model's doing:
  W291/W292/W293  trailing/blank-line whitespace (formatting noise)
  N802, UP004     `class Solution(object)` / camelCase method names come from
                  LiveBench's starter code, identical in both arms
  E501            line length
  F821            an undefined name the grader SUPPLIES: LCB prepends
                  `from typing import *`, `from collections import *` ... to
                  every program, so `List` or `defaultdict` are defined there
Also: ast.parse accepts `return` outside a function; the compiler does not.
The grader runs the compiled program, so "compiles" uses compile().

Also records, for completion answers, whether the model re-emitted the
starter (a second `class Solution`/`def <fn>` in the rebuilt program) instead
of continuing it.
"""
import builtins, json, os, re, subprocess, sys, tempfile, statistics as st
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import code_quality as cq  # noqa: E402

NOISE = {"W291", "W292", "W293", "N802", "UP004", "E501"}
IDIOM = cq.IDIOM
ns: dict = {}
exec(cq.IMPORT_STRING, ns)
HARNESS = set(ns) | set(dir(builtins))


def sign_p(pos, neg):
    n = pos + neg
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(pos, neg) + 1)) / 2 ** n)


def lint(code):
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(code); p = f.name
    try:
        r = subprocess.run([cq.PY, "-m", "ruff", "check", "--isolated", "--select", cq.RULES.replace("C90,", ""),
                            "--output-format", "json", "--target-version", "py311", p],
                           capture_output=True, text=True, timeout=60)
        items = json.loads(r.stdout or "[]")
    finally:
        os.unlink(p)
    keep = []
    for it in items:
        c = it.get("code") or "invalid-syntax"
        if c in NOISE:
            continue
        if c == "F821":
            m = re.search(r"`([^`]+)`", it.get("message", ""))
            if m and m.group(1) in HARNESS:
                continue
        keep.append((c, it.get("message", "")[:80]))
    return keep


def compiles(code):
    try:
        compile(code, "sol.py", "exec"); return True
    except SyntaxError as e:
        return f"{type(e).__name__}: {e.msg}"


qs = {q["question_id"]: q for q in json.load(open(os.path.join(HERE, "data", "lb_questions_slim.json")))}
out = []
for arm, fn in cq.ARMS.items():
    for a in json.load(open(os.path.join(HERE, "data", fn))):
        q = qs[a["question_id"]]
        ext, full = cq.rebuild(q, a["choices"][0]["turns"][-1])
        f = lint(full)
        fam = lambda c: "".join(ch for ch in c if ch.isalpha())
        reemit = None
        if q["task"] == "coding_completion":
            reemit = bool(re.search(r"^\s*class Solution\b", ext, re.M))
        out.append({"arm": arm, "id": a["question_id"], "task": q["task"], "compiles": compiles(full),
                    "attrib": len(f), "idiom": sum(1 for c, _ in f if fam(c) in IDIOM),
                    "correctness_class": sum(1 for c, _ in f if fam(c) in ("F", "B") or c == "invalid-syntax"),
                    "codes": [c for c, _ in f], "reemits_starter": reemit})
json.dump(out, open(os.path.join(HERE, "data", "code_quality_refined.json"), "w"), indent=1)
by = {(r["arm"], r["id"]): r for r in out}
ids = sorted({r["id"] for r in out})
print("does not compile (rebuilt as graded):")
for arm in cq.ARMS:
    print(f"  {arm:15}", [(i[:8], by[(arm, i)]["compiles"]) for i in ids if by[(arm, i)]["compiles"] is not True])
print("completion answers that re-emit `class Solution` instead of continuing:")
for arm in cq.ARMS:
    sub = [i for i in ids if by[(arm, i)]["task"] == "coding_completion"]
    print(f"  {arm:15} {sum(1 for i in sub if by[(arm, i)]['reemits_starter'])}/{len(sub)}")
for cls in ("all", "LCB_generation", "coding_completion"):
    sub = [i for i in ids if cls == "all" or by[("bonsai", i)]["task"] == cls]
    print(f"-- {cls} n={len(sub)}")
    for m in ("attrib", "idiom", "correctness_class"):
        a = [by[("bonsai", i)][m] for i in sub]; b = [by[("yamadori-xhigh", i)][m] for i in sub]
        up = sum(1 for x, y in zip(a, b) if y > x); dn = sum(1 for x, y in zip(a, b) if y < x)
        print(f"   {m:17} bonsai sum {sum(a):3} median {st.median(a):4}  xhigh sum {sum(b):3} median {st.median(b):4}  xhigh more {up} fewer {dn} same {len(sub)-up-dn}  sign p={sign_p(up, dn):.3f}")
for arm in cq.ARMS:
    c = {}
    for i in ids:
        for k in by[(arm, i)]["codes"]:
            c[k] = c.get(k, 0) + 1
    print(f"  {arm:15} codes: {sorted(c.items(), key=lambda kv: -kv[1])}")
