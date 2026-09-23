#!/usr/bin/env python
"""LiveCodeBench against the proxy, as a model, versus the model alone.

WHY THIS BENCHMARK, WHICH I FIRST SAID WAS THE WRONG ONE

The objection I raised was that LiveCodeBench problems are self-contained --
no repository, no imports, nothing to retrieve -- so the retrieval half of this
stack can do nothing on them.

That is true of the retrieval half and irrelevant to the rest. Fan-out,
selection, the concept seed, the reasoning budget and the max_tokens floor all
change the distribution the sampler draws from, on ANY generation task. And
from a client's side the proxy IS the model: one OpenAI endpoint. So pointing
the identical harness at :1234 instead of :11434 measures the hybrid exactly as
shipped, with nothing adapted to make it look good.

The retrieval tools no-opping here is a feature. It isolates the generation
mechanisms, which is the only way to attribute an effect to them.

WHAT MAKES THE COMPARISON FAIR

Paired. Every problem is run under both conditions, in the same process, with
the same sampling parameters, and scored by executing the same tests. Problem
difficulty varies enormously, and comparing two conditions on two different
samples would measure the samples. McNemar's exact test then looks only at the
problems where the conditions disagreed, which are the only ones carrying
information about the difference.

CONTAMINATION

Release v6 covers contests from 2025-01-04 to 2025-04-06. That is the whole
point of LiveCodeBench's date partitioning and it is the one guarantee three.js
could never give us -- three.js is in every training set, and every number
measured against it in this repo is labelled contaminated for that reason.

EXECUTION

Generated code is run in a subprocess, under a temporary working directory,
with a wall-clock timeout. It is model-written code solving algorithm problems,
not something hostile, but it is also not code anyone reviewed, so it does not
run in this interpreter.

RESUMABLE

Results append to jsonl as they finish and completed (problem, condition)
pairs are skipped on a re-run. A full pass is hours of generation on one GPU,
and a run that cannot be interrupted is a run that never finishes.
"""
from __future__ import annotations

import argparse
import ast
import base64
import json
import math
import os
import pickle
import random
import re
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.request
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import quality  # noqa: E402

# Two ways to run the comparison, and the second is cleaner.
#
# ENDPOINTS (the original) contrasts the bare model server against the proxy.
# That confounds what the augmentations do with what the PROXY does -- its
# preamble, its injected system block, its tool merge -- so a difference
# cannot be attributed to either.
#
# TIERS contrasts two `reasoning_effort` values on the SAME endpoint, through
# the same code path, with the same preamble. Only the augmentation set
# differs, which is the thing under test. `minimal` is the raw model with
# nothing added; `max` is everything.
ENDPOINT_CONDITIONS = {
    "direct": {"url": "http://127.0.0.1:11434/v1", "model": "bonsai"},
    "proxy": {"url": "http://127.0.0.1:1234/v1", "model": "bonsai"},
}

# Three arms, because two different questions are being asked.
#
#   minimal vs aug_on   "does the product beat the raw model" -- the headline,
#                       but it moves reasoning AND augmentation together, so a
#                       win is not attributable to either.
#   aug_off vs aug_on   the same reasoning level, the same preamble, the same
#                       code path, differing ONLY in whether tools and hints
#                       are injected. This is the attributable contrast.
#
# Fan-out stays off in all three. It multiplies generation cost and would
# dominate any difference; it gets its own experiment.
_PROXY = "http://127.0.0.1:1234/v1"
_OFF = '{"retrieval": false, "hints": false, "fanout": 1, "investigate": false}'
_ON = '{"retrieval": true, "hints": true, "fanout": 1, "investigate": false}'

TIER_CONDITIONS = {
    "minimal": {"url": _PROXY, "model": "yamadori",
                "extra": {"reasoning_effort": "minimal"}},
    "aug_off": {"url": _PROXY, "model": "yamadori",
                "extra": {"reasoning_effort": "medium"}, "features": _OFF},
    "aug_on": {"url": _PROXY, "model": "yamadori",
               "extra": {"reasoning_effort": "medium"}, "features": _ON},
}

CONDITIONS = dict(ENDPOINT_CONDITIONS)

PROMPT_STDIN = """You will be given a competitive programming question.
Write a complete Python 3 program that reads from standard input and writes to
standard output.

Return only the program, in a single ```python code block.

### Question
{question}
"""

PROMPT_FUNCTIONAL = """You will be given a competitive programming question.
Complete the given function. Do not read from standard input.

Return only the completed code, in a single ```python code block.

### Question
{question}

### Starter code
```python
{starter}
```
"""

_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)


def extract_code(text: str) -> str:
    """The last fenced block, or the whole reply if it is bare code.

    The last block, not the first: a model that explains its approach with a
    snippet and then gives the answer would otherwise be scored on the snippet.
    The proxy also prepends a one-line preamble on a first turn, which must not
    end up in the program.
    """
    blocks = _FENCE.findall(text or "")
    if blocks:
        return blocks[-1].strip()
    # No fence. Keep it only if it plausibly IS code, so an apology does not
    # get executed and scored as a wrong answer.
    #
    # MEASURED: this path scored a model failure that was ours. A reply that
    # hit the token cap never closed its fence, so extraction fell through to
    # here; the regex below matched `class Solution:` further down and returned
    # the WHOLE reply -- including the proxy's preamble on line 1:
    #
    #     sol.py", line 1
    #         `yamadori` - no repository detected in this conversation - ...
    #     SyntaxError: invalid character '·' (U+00B7)
    #
    # The docstring above already promised the preamble must not end up in the
    # program. It says so because someone foresaw this; it just was not done.
    # So drop leading lines until the first that looks like code, rather than
    # returning from the top of a reply whose first line is prose.
    t = (text or "").strip()
    lines = t.split("\n")
    start = next((i for i, ln in enumerate(lines)
                  if re.match(r"\s*(?:def|class|import|from|print|for|while|if|@)\b", ln)),
                 None)
    if start is None:
        return ""
    return "\n".join(lines[start:]).strip()


def private_cases(row: dict) -> list[dict]:
    raw = row.get("private_test_cases") or ""
    try:
        return json.loads(raw)
    except Exception:                                            # noqa: BLE001
        pass
    try:
        return json.loads(pickle.loads(zlib.decompress(
            base64.b64decode(raw.encode("utf-8")))))
    except Exception:                                            # noqa: BLE001
        return []


def all_cases(row: dict, cap: int) -> list[dict]:
    try:
        pub = json.loads(row.get("public_test_cases") or "[]")
    except Exception:                                            # noqa: BLE001
        pub = []
    cases = pub + private_cases(row)
    # Capped because some problems ship hundreds of cases and a wrong solution
    # would otherwise spend the timeout on every one of them. Public cases
    # come first so an obviously-broken program fails fast.
    return cases[:cap]


# LeetCode's judge pre-imports these. The starter code uses `List[int]` in its
# signatures and never imports typing, so a functional solution run standalone
# dies with `NameError: name 'List' is not defined` at the method signature --
# line 2, inside the class body, BEFORE any of the model's logic executes.
#
# Measured: this caused 3 of 4 crash failures, and 63 of the 175 problems in
# the v6 set are functional, so roughly a third of the benchmark was scoring
# the harness rather than the model. Every one was recorded as a model failure.
FUNCTIONAL_PREAMBLE = """import bisect, collections, functools, heapq, itertools
import math, random, re, string, sys
from collections import Counter, OrderedDict, defaultdict, deque
from functools import cache, lru_cache, reduce
from itertools import accumulate, combinations, permutations, product
from math import comb, gcd, inf, isqrt, lcm
from typing import Any, Dict, List, Optional, Set, Tuple, Union


class ListNode:
    def __init__(self, val=0, next=None):
        self.val, self.next = val, next


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val, self.left, self.right = val, left, right


"""


# Everything FUNCTIONAL_PREAMBLE binds, derived by executing it rather than
# listed by hand, so editing the preamble can never leave this stale.
PREAMBLE_NAMES = set()
_ns: dict = {}
exec(compile(FUNCTIONAL_PREAMBLE, "<preamble>", "exec"), _ns)   # noqa: S102
PREAMBLE_NAMES = {k for k in _ns if not k.startswith("__")}
del _ns


def _needs_preamble(code: str) -> bool:
    """Does this reply use a name the preamble supplies and never binds it?

    The old test was `"from typing import" in code`, a substring. A reply that
    imported only `Optional` satisfied it and then died at the method signature
    on `NameError: name 'List' is not defined` -- line 2, before any of the
    model's logic ran. Two such crashes were scored as model failures in the
    last run, which is the harness marking its own homework wrong.

    So ask the AST instead: which names does the code LOAD, and which does it
    actually BIND (import, def, class, assignment)? If it loads something the
    preamble provides and binds it nowhere, it needs the preamble.

    Unparseable code needs it too -- if the answer cannot be determined, the
    duplicate import is free and the crash is not.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return True
    bound, used = set(), set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            bound |= {(a.asname or a.name).split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom):
            bound |= {(a.asname or a.name) for a in n.names}
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(n.name)
        elif isinstance(n, ast.Name):
            (bound if isinstance(n.ctx, ast.Store) else used).add(n.id)
    return bool((used & PREAMBLE_NAMES) - bound)


def build_program(code: str, row: dict, case: dict) -> str:
    """The file that actually runs: the solution plus its harness."""
    if case.get("testtype") == "functional" or row.get("starter_code", "").strip():
        fn = (row.get("metadata") or {})
        if isinstance(fn, str):
            try:
                fn = json.loads(fn)
            except Exception:                                    # noqa: BLE001
                fn = {}
        name = fn.get("func_name") or ""
        args = case.get("input", "")
        # The preamble goes FIRST, and only when the model has not already
        # imported what it needs. A duplicate import is harmless, but
        # prepending unconditionally would shift every line number in the
        # traceback and make the next diagnosis harder than this one was.
        body = code if not _needs_preamble(code) else FUNCTIONAL_PREAMBLE + code
        return (f"{body}\n\n"
                f"import json, sys\n"
                f"_args = [__import__('ast').literal_eval(l) for l in "
                f"{args!r}.split(chr(10)) if l.strip()]\n"
                f"_r = Solution().{name}(*_args)\n"
                f"print(json.dumps(_r))\n")
    return code


def expected_matches(got: str, want: str, functional: bool) -> bool:
    """Compare output, tolerating formatting that does not change the answer."""
    got, want = (got or "").strip(), (want or "").strip()
    if got == want:
        return True
    if functional:
        try:
            return json.loads(got) == ast.literal_eval(want)
        except Exception:                                        # noqa: BLE001
            return False
    # stdin problems: ignore trailing whitespace per line and at the end
    a = [l.rstrip() for l in got.splitlines() if l.strip() != ""]
    b = [l.rstrip() for l in want.splitlines() if l.strip() != ""]
    if a == b:
        return True
    # numeric tolerance, for problems whose answer is a float
    if len(a) == len(b):
        try:
            return all(abs(float(x) - float(y)) <= 1e-6 * max(1.0, abs(float(y)))
                       for x, y in zip(a, b))
        except ValueError:
            return False
    return False


def _run_one(path: str, stdin_text: str, timeout: float,
             cwd: str) -> tuple[int, str, str, float, int]:
    """Run the program, returning (rc, stdout, stderr, seconds, peak bytes).

    Peak resident memory is sampled rather than accounted exactly: Windows has
    no `resource.getrusage`, and psutil can only read a process that is still
    alive. A short program can finish between samples, so the figure is a
    LOWER BOUND on the peak and is reported as such. It is still enough to
    separate a solution holding the whole input in a list from one streaming
    it, which is the distinction that matters here.
    """
    import threading

    import psutil

    t0 = time.time()
    p = subprocess.Popen([sys.executable, path],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True, cwd=cwd)
    peak = [0]

    def watch() -> None:
        try:
            pr = psutil.Process(p.pid)
            while p.poll() is None:
                try:
                    peak[0] = max(peak[0], pr.memory_info().rss)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    return
                time.sleep(0.004)
        except psutil.Error:
            return

    t = threading.Thread(target=watch, daemon=True)
    t.start()
    try:
        out, err = p.communicate(input=stdin_text, timeout=timeout)
    except subprocess.TimeoutExpired:
        p.kill()
        p.communicate()
        raise
    return p.returncode, out, err, time.time() - t0, peak[0]


def run_tests(code: str, row: dict, cases: list[dict],
              timeout: float) -> tuple[bool, str, dict]:
    """True only if every case passes, plus what the program cost to run.

    Cost is collected even though LiveCodeBench scores pass/fail only. Among
    solutions that BOTH conditions get right, runtime and peak memory are the
    difference between an O(n log n) answer and an O(n^2) one that squeaked
    under the limit -- an objective quality signal, measured by executing the
    code we already execute, with no taste involved.

    Only the slowest case is kept. Competitive problems ship one adversarial
    case among many trivial ones, and the mean over all of them would hide it.
    """
    perf = {"max_s": 0.0, "total_s": 0.0, "peak_kb": 0}
    if not code.strip():
        return False, "no code in the reply", perf
    with tempfile.TemporaryDirectory() as tmp:
        for i, case in enumerate(cases):
            functional = (case.get("testtype") == "functional"
                          or bool(row.get("starter_code", "").strip()))
            prog = build_program(code, row, case)
            path = os.path.join(tmp, "sol.py")
            with open(path, "w", encoding="utf-8") as f:
                f.write(prog)
            try:
                rc, out, err, secs, peak = _run_one(
                    path, "" if functional else case.get("input", ""),
                    timeout, tmp)
            except subprocess.TimeoutExpired:
                return False, f"timeout on case {i}", perf
            except Exception as e:                               # noqa: BLE001
                return False, f"{type(e).__name__} on case {i}: {e}", perf
            perf["max_s"] = max(perf["max_s"], round(secs, 3))
            perf["total_s"] = round(perf["total_s"] + secs, 3)
            perf["peak_kb"] = max(perf["peak_kb"], peak // 1024)
            if rc != 0:
                return False, f"case {i} exited {rc}: {err[-200:]}", perf
            if not expected_matches(out, case.get("output", ""), functional):
                return False, f"case {i} wrong output", perf
    return True, "", perf


def api_key() -> str:
    """The proxy's key, from the environment or the local client config.

    The proxy authenticates; the model server behind it does not. The first
    smoke run sent no key, every proxy request came back 401, and the report
    scored that as pass@1 0.0% against direct's 66.7% -- a "result" that was
    entirely an unset header. Three problems of wasted generation caught what
    a full run would have presented as a finding.
    """
    key = os.environ.get("YAMADORI_API_KEY", "").strip()
    if key:
        return key
    cfg = os.path.join(os.environ.get("LOCALAPPDATA", ""), "hermes", "config.yaml")
    try:
        for line in open(cfg, encoding="utf-8"):
            if "api_key" in line and ":" in line:
                return line.split(":", 1)[1].strip().strip('"').strip("'")
    except OSError:
        pass
    return ""


_KEY = ""


def generate(cond: dict, prompt: str, max_tokens: int,
             timeout: int) -> tuple[str, dict]:
    body = {"model": cond["model"],
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0.2}
    # A tier arm carries its own request fields. Merged rather than replacing,
    # so max_tokens stays under the caller's control and the tier floor can
    # raise it server-side, which is where that decision belongs.
    body.update(cond.get("extra") or {})
    t0 = time.time()
    # Connection: close on every request.
    #
    # The proxy speaks HTTP/1.1, so urllib keeps the socket open and reuses
    # it for the next problem. Between two generations that can be minutes,
    # by which point the server has dropped it -- the reused socket then
    # fails and surfaces as a 502, which the harness records as a generation
    # error. Measured: 8 of 16 attempts failed that way, while the identical
    # request sent on a fresh connection succeeded every time.
    #
    # A new connection per generation costs a TCP handshake against a call
    # that takes 30 to 200 seconds. It is free.
    headers = {"Content-Type": "application/json", "Connection": "close"}
    if _KEY:
        headers["Authorization"] = f"Bearer {_KEY}"
    # Feature toggles ride on a header so the request body stays the OpenAI
    # schema and the override never becomes part of the cached prompt.
    if cond.get("features"):
        headers["X-Yamadori-Features"] = cond["features"]
    req = urllib.request.Request(f"{cond['url']}/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    # `incomplete` is the proxy saying the upstream connection died mid-answer
    # and handing back the part that arrived. That is a TRANSPORT fault, not a
    # model failure, and scoring the fragment would charge the model for our
    # plumbing -- which is exactly what produced a run where roughly 60% of
    # rows were errors misattributed to generation. Raised so the harness
    # records it as an error and leaves the row unfinished, to be retried.
    if (d["choices"][0].get("finish_reason") or "") == "incomplete":
        raise RuntimeError(
            "upstream dropped mid-generation; partial answer not scored")
    msg = d["choices"][0]["message"]
    text = msg.get("content") or ""
    if not text.strip() and msg.get("reasoning_content"):
        # The empty-content-with-full-reasoning failure this repo already
        # documents. Recorded rather than hidden, because whether it still
        # happens is part of what is being measured.
        text = msg["reasoning_content"]
    # Tokens as reported by the server, not estimated from characters.
    # Cost is half the comparison: an arm that wins by spending three times
    # the tokens has not obviously won, and without this the table could only
    # show latency, which conflates queueing with work.
    u = d.get("usage") or {}
    return text, {
        "ms": round((time.time() - t0) * 1000),
        "finish": (d["choices"][0].get("finish_reason") or ""),
        "empty_content": not (msg.get("content") or "").strip(),
        "chars": len(text),
        "prompt_tokens": u.get("prompt_tokens"),
        "completion_tokens": u.get("completion_tokens"),
        "total_tokens": u.get("total_tokens"),
    }


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


def load_done(path: str) -> set:
    """Completed pairs only. An ERROR is not a completion.

    The first version added every row, so a generation that failed because
    the server was restarting was never retried -- it counted as done
    forever. Three restarts during one run permanently poisoned 13 problems
    and left 3 of 45 complete. A transient failure must be retried, which
    means it must not be recorded as finished.
    """
    done = set()
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            try:
                r = json.loads(line)
            except Exception:                                    # noqa: BLE001
                continue
            if r.get("error"):
                continue
            done.add((r["question_id"], r["condition"]))
    return done


def report(path: str) -> None:
    rows = []
    for line in open(path, encoding="utf-8"):
        try:
            rows.append(json.loads(line))
        except Exception:                                        # noqa: BLE001
            continue
    by: dict = {}
    errs = [r for r in rows if r.get("error")]
    for r in rows:
        by.setdefault(r["question_id"], {})[r["condition"]] = r
    # A generation that never happened is not a failed solution. Scoring it as
    # one is how the first smoke run reported the proxy at 0.0% when every
    # request had been rejected for a missing header.
    paired = {q: v for q, v in by.items()
              if len(v) == len(CONDITIONS)
              and not any(x.get("error") for x in v.values())}
    if errs:
        kinds: dict = {}
        for r in errs:
            k = (r["condition"], r["error"].split(":")[0])
            kinds[k] = kinds.get(k, 0) + 1
        print(f"\n  {len(errs)} generation errors, excluded from scoring:")
        for (c, k), v in sorted(kinds.items(), key=lambda x: -x[1]):
            print(f"    {c:<8} {k:<24} {v}")

    print(f"\n{'=' * 74}")
    print(f"  LIVECODEBENCH v6  --  {len(paired)} problems complete under "
          f"both conditions")
    print(f"  (of {len(by)} attempted; contests 2025-01-04 to 2025-04-06)")
    print(f"{'=' * 74}")
    if not paired:
        print("  nothing paired yet")
        return

    n = len(paired)
    print(f"  {'condition':<12}{'pass@1':<22}{'mean s':<9}"
          f"{'tok in':<9}{'tok out':<9}{'empty':<7}{'len-cut':<7}")
    print("  " + "-" * 62)
    for c in CONDITIONS:
        k = sum(1 for v in paired.values() if v[c]["passed"])
        lo, hi = wilson(k, n)
        secs = sum(v[c]["ms"] for v in paired.values()) / n / 1000
        empty = sum(1 for v in paired.values() if v[c].get("empty_content"))
        cut = sum(1 for v in paired.values() if v[c].get("finish") == "length")
        pin = [v[c].get("prompt_tokens") for v in paired.values()
               if v[c].get("prompt_tokens")]
        pout = [v[c].get("completion_tokens") for v in paired.values()
                if v[c].get("completion_tokens")]
        mi = round(sum(pin) / len(pin)) if pin else 0
        mo = round(sum(pout) / len(pout)) if pout else 0
        print(f"  {c:<12}{k}/{n} = {k / n:5.1%} [{lo:.0%}-{hi:.0%}]".ljust(34)
              + f"{secs:<9.1f}{mi:<9}{mo:<9}{empty:<7}{cut:<7}")

    names = list(CONDITIONS)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b_ = names[i], names[j]
            b = sum(1 for v in paired.values()
                    if v[b_]["passed"] and not v[a]["passed"])
            c = sum(1 for v in paired.values()
                    if v[a]["passed"] and not v[b_]["passed"])
            p = mcnemar(b, c)
            verdict = (f"{b_} better" if b > c and p < 0.05 else
                       f"{a} better" if c > b and p < 0.05 else
                       "no detectable difference")
            print(f"\n  McNemar {a} vs {b_}: {b_} solved {b} that {a} missed, "
                  f"{a} solved {c} that {b_} missed")
            print(f"    p = {p:.4f}  ->  {verdict}")
            if b + c < 10:
                print(f"    only {b + c} discordant pairs: this cannot detect "
                      f"anything but a large effect yet")

    # Among problems BOTH arms solved, is one arm's code simpler or faster?
    # Restricted to problems both passed, because comparing the complexity of
    # a right answer against a wrong one measures nothing.
    a, b_ = names[0], names[1]
    both = [v for v in paired.values() if v[a]["passed"] and v[b_]["passed"]]
    if both:
        print(f"\n  among {len(both)} problems BOTH conditions solved:")
        print(f"    {'metric':<16}{a:>12}{b_:>12}   sign test")
        for key, label in (("max_s", "slowest case s"),
                           ("peak_kb", "peak KB"),
                           ("q_cyclomatic", "cyclomatic"),
                           ("q_statements", "statements"),
                           ("q_max_depth", "nesting depth")):
            pairs = [(v[a].get(key), v[b_].get(key)) for v in both
                     if v[a].get(key) is not None and v[b_].get(key) is not None]
            if len(pairs) < 3:
                continue
            ma = statistics.median(p[0] for p in pairs)
            mb = statistics.median(p[1] for p in pairs)
            st = quality.sign_test(pairs)
            verdict = (f"{b_} better" if st["better"] > st["worse"] and st["p"] < 0.05
                       else f"{a} better" if st["worse"] > st["better"] and st["p"] < 0.05
                       else "no difference")
            print(f"    {label:<16}{ma:>12.2f}{mb:>12.2f}   "
                  f"{st['better']}-{st['worse']} p={st['p']:.3f} {verdict}")
        print("    (sign test on the same problem; ties within 5% ignored)")

    print("\n  by difficulty")
    for diff in ("easy", "medium", "hard"):
        g = {q: v for q, v in paired.items()
             if v[names[0]].get("difficulty") == diff}
        if not g:
            continue
        cells = "".join(
            f"{sum(1 for v in g.values() if v[c]['passed']) / len(g):>10.1%}"
            for c in CONDITIONS)
        print(f"    {diff:<8} n={len(g):<4}{cells}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "data", "test6.jsonl"))
    ap.add_argument("--n", type=int, default=45, help="problems, stratified")
    ap.add_argument("--max-tokens", type=int, default=8000)
    ap.add_argument("--gen-timeout", type=int, default=1800)
    ap.add_argument("--test-timeout", type=float, default=12.0)
    ap.add_argument("--max-cases", type=int, default=12)
    ap.add_argument("--seed", type=int, default=20260921)
    ap.add_argument("--out", default=os.path.join(HERE, "lcb_results.jsonl"))
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--tiers", action="store_true",
                    help="contrast reasoning_effort tiers on ONE endpoint "
                         "instead of two endpoints; isolates the augmentations "
                         "from the proxy's own overhead")
    args = ap.parse_args()

    global CONDITIONS
    if args.tiers:
        CONDITIONS = dict(TIER_CONDITIONS)

    if args.report_only:
        report(args.out)
        return

    global _KEY
    _KEY = api_key()
    # Prove BOTH endpoints answer before spending hours on them -- rule 1.
    for name, cond in CONDITIONS.items():
        try:
            txt, meta = generate(cond, "Reply with the word ready.", 64, 120)
            print(f"  {name:<8} alive, {meta['ms']}ms, {txt.strip()[:40]!r}")
        except Exception as e:                                   # noqa: BLE001
            raise SystemExit(
                f"\n  {name} ({cond['url']}) is not answering: "
                f"{type(e).__name__}: {e}\n"
                f"  Refusing to run. A dead endpoint scores 0% and reads as a "
                f"result.")

    rows = [json.loads(l) for l in open(args.data, encoding="utf-8")]
    rnd = random.Random(args.seed)
    # Stratified, so a short run is not accidentally all easy problems.
    picked = []
    per = max(1, args.n // 3)
    for diff in ("easy", "medium", "hard"):
        g = [r for r in rows if r.get("difficulty") == diff]
        rnd.shuffle(g)
        picked += g[:per]
    rnd.shuffle(picked)

    done = load_done(args.out)
    print(f"  {len(picked)} problems selected, {len(done)} (problem, condition) "
          f"pairs already done")

    with open(args.out, "a", encoding="utf-8") as out:
        for i, row in enumerate(picked):
            qid = row["question_id"]
            functional = bool(row.get("starter_code", "").strip())
            prompt = (PROMPT_FUNCTIONAL.format(question=row["question_content"],
                                               starter=row["starter_code"])
                      if functional else
                      PROMPT_STDIN.format(question=row["question_content"]))
            cases = all_cases(row, args.max_cases)
            for cname, cond in CONDITIONS.items():
                if (qid, cname) in done:
                    continue
                try:
                    text, meta = generate(cond, prompt, args.max_tokens,
                                          args.gen_timeout)
                except Exception as e:                           # noqa: BLE001
                    rec = {"question_id": qid, "condition": cname,
                           "difficulty": row.get("difficulty"),
                           "passed": False, "ms": 0,
                           "error": f"{type(e).__name__}: {e}"}
                    out.write(json.dumps(rec) + "\n")
                    out.flush()
                    print(f"  [{i + 1}/{len(picked)}] {qid:<14} {cname:<7} "
                          f"GENERATION FAILED {type(e).__name__}", flush=True)
                    continue
                code = extract_code(text)
                passed, why, perf = run_tests(code, row, cases,
                                              args.test_timeout)
                # Structural metrics on the code itself. Correctness is the
                # veto; among solutions that all pass, what separates them is
                # speed, memory and how the code reads. Recorded per arm so
                # the comparison is arm-against-arm on the SAME problem, which
                # needs no reference solution and cannot be contaminated.
                m = quality.metrics(code) or {}
                rec = {"question_id": qid, "condition": cname,
                       "difficulty": row.get("difficulty"),
                       "passed": passed, "why": why, "n_cases": len(cases),
                       "parsed": bool(m),
                       **{f"q_{k}": v for k, v in m.items()},
                       **perf, **meta}
                out.write(json.dumps(rec) + "\n")
                out.flush()
                print(f"  [{i + 1}/{len(picked)}] {qid:<14} {cname:<7} "
                      f"{'PASS' if passed else 'fail'} "
                      f"{meta['ms'] / 1000:5.1f}s  {why[:44]}", flush=True)

    report(args.out)


if __name__ == "__main__":
    main()
