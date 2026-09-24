"""Q2: does the code look better with the stack? Offline, paired per question.

Arms: bonsai (all augmentation forced off, effort medium) vs yamadori-xhigh
(all allowed, effort medium), LiveBench release 2024-11-25, the 21 coding
questions of run lb-20260923-minp0. The answers are the ones LiveBench scored
(dumped from the WSL harness by dump_livebench.py).

Programs are rebuilt exactly as LiveBench's grader does
(livebench/process_results/coding/utils.py LCB_generation_process_results):
extract_code (last fenced block), then, for a completion question, the starter
`partial_solution` + "\\n" + the extraction unless it already starts with it.

Per program:
  ruff      findings with a wide rule set, grouped by family (idiom families:
            SIM, C4, UP, PERF, RUF, PL, B, RET; plus E/W/F/N/ARG); C901 at
            max-complexity 0 gives McCabe per function
  ast_cc    1 + decision points over the WHOLE program (module-level stdin
            scripts included, which C901 does not see)
  length    non-blank, non-comment lines and characters of the model-written
            part (the extraction; the starter is identical in both arms)
  runtime   every public test, and every private test, run in a child process
            with a real stdin/stdout (both with `.buffer`, so the grader quirk
            is excluded from timing), each test timed around the call/exec
            only; public: min of 5, private: min of 3 (1 if a run takes >1 s)

Nothing here calls a model or a service. Writes data/code_quality.json.
"""
from __future__ import annotations

import ast
import json
import os
import pickle
import base64
import subprocess
import sys
import tempfile
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
PY = sys.executable
BIG = os.environ.get("LB_QUESTIONS_FULL")  # the dump with private tests
ARMS = {"bonsai": "answers_yamadori-bonsai-arm.json",
        "yamadori-xhigh": "answers_yamadori-tier-xhigh.json"}
RULES = "E,W,F,C90,N,B,SIM,C4,UP,PERF,RUF,PL,RET,ARG"
IDIOM = ("SIM", "C4", "UP", "PERF", "RUF", "PL", "B", "RET")

IMPORT_STRING = ("from string import *\nfrom re import *\nfrom datetime import *\n"
                 "from collections import *\nfrom heapq import *\nfrom bisect import *\n"
                 "from copy import *\nfrom math import *\nfrom random import *\n"
                 "from statistics import *\nfrom itertools import *\nfrom functools import *\n"
                 "from operator import *\nfrom io import *\nfrom sys import *\n"
                 "from json import *\nfrom builtins import *\nfrom typing import *\n"
                 "import string\nimport re\nimport datetime\nimport collections\nimport heapq\n"
                 "import bisect\nimport copy\nimport math\nimport random\nimport statistics\n"
                 "import itertools\nimport functools\nimport operator\nimport io\nimport sys\n"
                 "import json\nsys.setrecursionlimit(50000)\n")


def extract_code(model_output: str) -> str:
    """livebench.lcb_runner.utils.extraction_utils.extract_code, lmstyle None."""
    outputlines = model_output.rstrip().split("\n")
    indexlines = [i for i, line in enumerate(outputlines) if "```" in line]
    if len(indexlines) < 2:
        if len(model_output) > 1 and model_output[0] == "`" and model_output[-1] == "`":
            return model_output[1:-1]
        if len(indexlines) == 1 and indexlines[0] == len(outputlines) - 1:
            return "\n".join(outputlines[:-1])
        return model_output.rstrip()
    return "\n".join(outputlines[indexlines[-2] + 1: indexlines[-1]])


def rebuild(q: dict, answer: str) -> tuple[str, str]:
    ext = extract_code(answer)
    ps = q.get("partial_solution")
    if ps and not ext.startswith(ps):
        return ext, ps + "\n" + ext
    return ext, ext


def sloc(code: str) -> int:
    return sum(1 for ln in code.splitlines()
               if ln.strip() and not ln.strip().startswith("#"))


def ast_cc(code: str) -> int | None:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    n = 1
    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.For, ast.While, ast.AsyncFor,
                             ast.IfExp, ast.ExceptHandler, ast.Assert)):
            n += 1
        elif isinstance(node, ast.BoolOp):
            n += len(node.values) - 1
        elif isinstance(node, ast.comprehension):
            n += 1 + len(node.ifs)
        elif isinstance(node, ast.match_case):
            n += 1
    return n


def ruff(code: str) -> dict:
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False,
                                     encoding="utf-8") as f:
        f.write(code)
        path = f.name
    try:
        r = subprocess.run([PY, "-m", "ruff", "check", "--isolated",
                            "--select", RULES, "--output-format", "json",
                            "--config", "lint.mccabe.max-complexity = 0",
                            "--target-version", "py311", path],
                           capture_output=True, text=True, timeout=60)
        items = json.loads(r.stdout or "[]")
    finally:
        os.unlink(path)
    fams: dict[str, int] = {}
    codes: dict[str, int] = {}
    mccabe = []
    for it in items:
        c = it.get("code") or "SYNTAX"
        if c == "C901":
            try:
                mccabe.append(int(it["message"].split("(")[1].split(" ")[0]))
            except (IndexError, ValueError):
                pass
            continue
        fam = "".join(ch for ch in c if ch.isalpha())
        fams[fam] = fams.get(fam, 0) + 1
        codes[c] = codes.get(c, 0) + 1
    total = sum(fams.values())
    return {"total": total, "idiom": sum(v for k, v in fams.items() if k in IDIOM),
            "families": fams, "codes": codes,
            "mccabe_max": max(mccabe) if mccabe else None,
            "mccabe_sum": sum(mccabe) if mccabe else None,
            "syntax_error": any((it.get("code") is None) for it in items)}


CHILD = r'''
import io, json, sys, time
spec = json.load(open(sys.argv[1], encoding="utf-8"))
code, t, reps = spec["code"], spec["test"], spec["reps"]
best, ok, err = None, None, None
for _ in range(reps):
    if t["testtype"] == "functional":
        ns = {"__name__": "sol"}
        try:
            exec(compile(spec["prefix"] + code, "sol.py", "exec"), ns)
            m = getattr(ns["Solution"](), spec["fn"])
            args = [json.loads(x) for x in t["input"].split("\n")]
            t0 = time.perf_counter(); out = m(*args); dt = time.perf_counter() - t0
            if isinstance(out, tuple): out = list(out)
            ok = (json.loads(t["output"]) == out)
        except BaseException as e:
            err = f"{type(e).__name__}: {str(e)[:120]}"; break
    else:
        fin = io.TextIOWrapper(io.BytesIO(t["input"].encode()), encoding="utf-8")
        buf = io.BytesIO(); fout = io.TextIOWrapper(buf, encoding="utf-8")
        old_in, old_out = sys.stdin, sys.stdout
        sys.stdin, sys.stdout = fin, fout
        try:
            t0 = time.perf_counter()
            exec(compile(code, "sol.py", "exec"), {"__name__": "__main__"})
            dt = time.perf_counter() - t0
        except SystemExit:
            dt = time.perf_counter() - t0
        except BaseException as e:
            sys.stdin, sys.stdout = old_in, old_out
            err = f"{type(e).__name__}: {str(e)[:120]}"; break
        finally:
            try: fout.flush()
            except Exception: pass
            sys.stdin, sys.stdout = old_in, old_out
        got = [x.strip() for x in buf.getvalue().decode("utf-8", "replace").strip().splitlines()]
        exp = [x.strip() for x in t["output"].strip().splitlines()]
        ok = got == exp
    best = dt if best is None else min(best, dt)
    if dt > 1.0: break
print(json.dumps({"s": best, "ok": ok, "err": err}))
'''


def run_test(code: str, test: dict, fn: str | None, reps: int, timeout: float) -> dict:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False,
                                     encoding="utf-8") as f:
        json.dump({"code": code, "test": test, "fn": fn, "reps": reps,
                   "prefix": IMPORT_STRING + "\n\n"}, f)
        path = f.name
    try:
        r = subprocess.run([PY, "-c", CHILD, path], capture_output=True,
                           text=True, timeout=timeout)
        line = (r.stdout.strip().splitlines() or ["{}"])[-1]
        try:
            return json.loads(line)
        except ValueError:
            return {"s": None, "ok": None, "err": (r.stderr or "no output")[-200:]}
    except subprocess.TimeoutExpired:
        return {"s": None, "ok": False, "err": f"timeout {timeout}s"}
    finally:
        os.unlink(path)


def private_tests(qid: str, big: dict) -> list[dict]:
    q = big.get(qid)
    if not q:
        return []
    raw = q["private_test_cases"]
    try:
        return json.loads(raw)
    except ValueError:
        return json.loads(pickle.loads(zlib.decompress(base64.b64decode(raw.encode("utf-8")))))


def main():
    qs = {q["question_id"]: q for q in json.load(open(os.path.join(HERE, "data", "lb_questions_slim.json")))}
    big = {}
    if BIG and os.path.exists(BIG):
        big = {q["question_id"]: q for q in json.load(open(BIG))}
    rows = {}
    for line in open(os.path.join(ROOT, "bench", "livebench", "results", "lb-20260923-minp0",
                                  "rows_yamadori-xhigh_coding.jsonl"), encoding="utf-8"):
        r = json.loads(line); rows[("yamadori-xhigh", r["id"])] = r
    for line in open(os.path.join(ROOT, "bench", "livebench", "results", "lb-20260923-minp0",
                                  "rows_bonsai_coding.jsonl"), encoding="utf-8"):
        r = json.loads(line); rows[("bonsai", r["id"])] = r
    out = []
    for arm, fn in ARMS.items():
        for a in json.load(open(os.path.join(HERE, "data", fn))):
            qid = a["question_id"]; q = qs[qid]
            text = a["choices"][0]["turns"][-1]
            ext, full = rebuild(q, text)
            meta = json.loads(q["original_json"]["metadata"]) if isinstance(q["original_json"], dict) else {}
            fname = meta.get("func_name")
            rec = {"arm": arm, "id": qid, "task": q["task"], "title": q.get("question_title"),
                   "score": rows[(arm, qid)]["score"],
                   "sloc": sloc(ext), "chars": len(ext),
                   "ast_cc": ast_cc(full), "ruff": ruff(full),
                   "buffer": ("stdin.buffer" in full or "stdout.buffer" in full),
                   "testtype": None, "public": [], "private": []}
            pub = json.loads(q["public_test_cases"])
            rec["testtype"] = pub[0]["testtype"] if pub else None
            for t in pub:
                rec["public"].append(run_test(full, t, fname, 5, 60))
            for t in private_tests(qid, big):
                rec["private"].append(run_test(full, t, fname, 3, 30))
            out.append(rec)
            pt = sum(x["s"] or 0 for x in rec["public"])
            print(f"{arm:15} {qid[:8]} {q['task'][:10]:10} score={rec['score']} sloc={rec['sloc']:3} "
                  f"cc={rec['ast_cc']} ruff={rec['ruff']['total']} idiom={rec['ruff']['idiom']} "
                  f"pub_ok={sum(1 for x in rec['public'] if x['ok'])}/{len(pub)} pub_s={pt:.5f} "
                  f"priv_ok={sum(1 for x in rec['private'] if x['ok'])}/{len(rec['private'])}", flush=True)
    json.dump(out, open(os.path.join(HERE, "data", "code_quality.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
