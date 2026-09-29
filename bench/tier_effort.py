"""Each tier model's best SINGLE effort, measured (operator, 2026-09-29: "These models seem to do better on one
reasoning level, so we pick the best reasoning level and capability is now the model swap").

A SMALL PAIRED SET, through :1234 (the door users use), one GPU consumer at a time:

  tasks   TASKS below: n=6 self-contained Python functions, each graded by EXECUTING its asserts (a subprocess, 20 s);
          written for this script (not from any benchmark, so not in any training set verbatim).
  arms    for each model (the tier that serves it: bonsai at medium, mirai-s at xhigh, flash-next at max) x each
          effort the served template accepts (low / medium / xhigh), the SAME tasks and the SAME seed per task.
  held    X-Yamadori-Features {"effort": E, "investigate": false, "fanout": 1, "repair": false, "check_code": false,
          "retrieval": false}: only the model and the effort vary -- no plan, no second brain, no injections.
  NO CAP  (operator, 2026-09-29: "they need to be based on what makes them perform the best and how much memory we
          have, so if we cap them and it gives us bad signal it poisons the entire point of offering more capable
          models"): every run thinks within its model's WINDOW only (the table's profile: thinking = main_cap -
          prompt - answer), so each generation's NATURAL thinking length is what gets recorded. --cap exists only to
          reproduce a capped grid on purpose; it is never the default.
  records per run: pass (asserts ran clean), finish_reason, reasoning / content / completion tokens, the thinking
          budget sent, wall seconds. Per arm: passes/n, median and max natural thinking, finish=length count, median
          seconds. THE CURVE (summary.json `curve`): per model, the passes at each effort and, at the best effort,
          the longest natural thinking among its passes -- a cap at or above it keeps every pass (the evidence a cap
          would need; none is chosen here). Written to OUT/effort.jsonl and OUT/summary.json.

A verdict is PAIRED (the same tasks): the best effort is the one with the most passes; a tie goes to fewer median
completion tokens. With n=6 a one-task difference is not a result (AGENTS.md: "If a measurement does not survive a
repeat, it is not a result"): the summary says so, and --reps repeats every run with another seed.

    python bench/tier_effort.py --key-file PATH --out DIR [--models mirai-s,flash-next] [--reps 1]
    python bench/tier_effort.py --selftest        # the grader and the code extraction, offline
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

PROXY = os.environ.get("YAMADORI_PROXY", "http://127.0.0.1:1234")
MODEL_TIER = {"bonsai": "medium", "mirai-s": "xhigh", "flash-next": "max"}
EFFORTS = ("low", "medium", "xhigh")
CAP = None          # no cap (operator 2026-09-29); --cap N reproduces a capped grid only on purpose
SEED = 1234

TASKS = [
    {"id": "interval_merge",
     "prompt": "Write a Python function `merge(intervals)` that takes a list of [start, end] integer pairs (in any "
               "order, possibly overlapping or touching) and returns the merged list sorted by start. Touching "
               "intervals like [1,2] and [2,3] merge. Reply with the function in one ```python block.",
     "tests": "assert merge([[1,3],[2,6],[8,10],[15,18]])==[[1,6],[8,10],[15,18]]\n"
              "assert merge([[1,4],[4,5]])==[[1,5]]\nassert merge([])==[]\n"
              "assert merge([[5,7],[1,2],[2,4]])==[[1,4],[5,7]]\nassert merge([[1,10],[2,3]])==[[1,10]]\n"},
    {"id": "roman",
     "prompt": "Write Python functions `to_roman(n)` (1 <= n <= 3999) and `from_roman(s)` converting between integers "
               "and standard Roman numerals (subtractive notation). Reply with both in one ```python block.",
     "tests": "assert to_roman(1994)=='MCMXCIV'\nassert to_roman(3999)=='MMMCMXCIX'\nassert to_roman(4)=='IV'\n"
              "assert all(from_roman(to_roman(i))==i for i in range(1,4000))\nassert from_roman('XLII')==42\n"},
    {"id": "lru",
     "prompt": "Write a Python class `LRU(capacity)` with `get(key)` (returns the value or -1, and marks the key most "
               "recently used) and `put(key, value)` (inserts or updates; when over capacity, evicts the least "
               "recently used key). Both O(1). Reply with the class in one ```python block.",
     "tests": "c=LRU(2)\nc.put(1,1);c.put(2,2)\nassert c.get(1)==1\nc.put(3,3)\nassert c.get(2)==-1\n"
              "c.put(4,4)\nassert c.get(1)==-1 and c.get(3)==3 and c.get(4)==4\n"
              "d=LRU(1);d.put(1,1);d.put(1,5);assert d.get(1)==5;d.put(2,2);assert d.get(1)==-1\n"},
    {"id": "expr_eval",
     "prompt": "Write a Python function `evaluate(expr)` that evaluates an arithmetic expression string with "
               "non-negative integers, + - * /, parentheses, spaces and unary minus, where / is integer division "
               "truncating toward zero. Do not use eval or exec. Reply with the function in one ```python block.",
     "tests": "assert evaluate('1 + 2 * 3')==7\nassert evaluate('(1+2)*3')==9\nassert evaluate('7/2')==3\n"
              "assert evaluate('-7/2')==-3\nassert evaluate('2*(3+4)-5/(1+1)')==12\nassert evaluate('-(2+3)*2')==-10\n"
              "assert evaluate(' 10 - 2 - 3 ')==5\n"},
    {"id": "topo_cycle",
     "prompt": "Write a Python function `order(deps)` where deps maps each task name to a list of task names it "
               "depends on. Return a list of all tasks (including ones that only appear as dependencies) such "
               "that every task comes after its dependencies; among ready tasks pick the alphabetically smallest "
               "first. If there is a cycle, raise ValueError. Reply with the function in one ```python block.",
     "tests": "assert order({'b':['a'],'c':['b','a']})==['a','b','c']\n"
              "assert order({'x':[],'a':[]})==['a','x']\n"
              "assert order({'d':['c'],'b':[],'c':['b','a']})==['a','b','c','d']\n"
              "try:\n    order({'a':['b'],'b':['a']})\n    raise SystemExit(1)\nexcept ValueError:\n    pass\n"},
    {"id": "run_length",
     "prompt": "Write Python functions `encode(s)` and `decode(s)` for run-length encoding where each run is written "
               "as its count followed by the character, e.g. 'aaab' -> '3a1b'. Characters may be any non-digit "
               "character; counts may exceed 9. decode(encode(s)) must equal s for every string without digits. "
               "Reply with both in one ```python block.",
     "tests": "assert encode('aaab')=='3a1b'\nassert decode('12x1y')=='x'*12+'y'\nassert encode('')==''\n"
              "import random\nrandom.seed(0)\nfor _ in range(200):\n"
              "    s=''.join(random.choice('ab c!') for _ in range(random.randint(0,40)))\n"
              "    assert decode(encode(s))==s\n"},
]


def extract(text: str) -> str:
    """The last ```python block (else the last ``` block, else the text)."""
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text or "", re.S)
    return blocks[-1] if blocks else (text or "")


def grade(code: str, tests: str) -> tuple[bool, str]:
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.py")
        with open(p, "w", encoding="utf-8") as f:
            f.write(code + "\n\n" + tests)
        try:
            r = subprocess.run([sys.executable, p], capture_output=True, text=True, timeout=20, cwd=d)
        except subprocess.TimeoutExpired:
            return False, "timeout"
        return r.returncode == 0, (r.stderr or "")[-300:]


def ask(key: str, task: dict, model: str, effort: str, cap: int, seed: int) -> dict:
    tier = MODEL_TIER[model]
    feats = {"effort": effort, "investigate": False, "fanout": 1, "repair": False, "check_code": False,
             "retrieval": False}
    if cap:
        feats["reasoning_cap"] = int(cap)
    body = {"model": "yamadori", "reasoning_effort": tier, "seed": seed,
            "messages": [{"role": "user", "content": task["prompt"]}]}
    req = urllib.request.Request(f"{PROXY}/v1/chat/completions", data=json.dumps(body).encode(), headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json",
        "X-Yamadori-Features": json.dumps(feats)})
    t0 = time.time()
    try:
        # not a thinking cap: a transport bound above the memory-bound worst case -- flash-next's whole window,
        # 259,072 tokens, at ~20 tok/s is ~3.6 h -- so no natural generation is cut by it
        with urllib.request.urlopen(req, timeout=4 * 3600) as r:
            d = json.load(r)
    except urllib.error.HTTPError as e:
        return {"status": e.code, "error": e.read().decode("utf-8", "replace")[:300], "s": round(time.time() - t0, 1)}
    ch = (d.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    u = d.get("usage") or {}
    x = d.get("x_yamadori") or {}
    ok, err = grade(extract(msg.get("content") or ""), task["tests"])
    comp = u.get("completion_tokens")
    reas = (u.get("completion_tokens_details") or {}).get("reasoning_tokens")
    return {"status": 200, "pass": ok, "err": err if not ok else "", "finish": ch.get("finish_reason"),
            "completion_tokens": comp, "reasoning_tokens": reas,
            "content_tokens": (comp - reas) if isinstance(comp, int) and isinstance(reas, int) else None,
            "thinking_budget_sent": (x.get("budget") or {}).get("reasoning_budget_tokens"),
            "s": round(time.time() - t0, 1), "served_by": (x.get("capacity") or {}).get("model"),
            "effort_sent": ((x.get("sampling") or {}).get("profile") or {}).get("effort", {}).get("sent")}


def summarise(rows: list[dict]) -> dict:
    out: dict = {}
    for r in rows:
        a = out.setdefault(r["model"], {}).setdefault(r["effort"], {"n": 0, "pass": 0, "tokens": [], "s": [],
                                                                    "think": [], "think_pass": [],
                                                                    "length": 0, "errors": 0})
        a["n"] += 1
        if r.get("status") != 200:
            a["errors"] += 1
            continue
        a["pass"] += bool(r.get("pass"))
        a["length"] += r.get("finish") == "length"
        a["tokens"].append(r.get("completion_tokens") or 0)
        a["s"].append(r.get("s") or 0)
        if isinstance(r.get("reasoning_tokens"), int):
            a["think"].append(r["reasoning_tokens"])
            if r.get("pass"):
                a["think_pass"].append(r["reasoning_tokens"])
    verdict = {}
    for m, arms in out.items():
        for e, a in arms.items():
            a["median_tokens"] = statistics.median(a["tokens"]) if a["tokens"] else None
            a["median_s"] = statistics.median(a["s"]) if a["s"] else None
            a["median_thinking"] = statistics.median(a["think"]) if a["think"] else None
            a["max_thinking"] = max(a["think"]) if a["think"] else None
            a["max_thinking_of_passes"] = max(a["think_pass"]) if a["think_pass"] else None
            del a["tokens"], a["s"], a["think"], a["think_pass"]
        best = sorted(arms.items(), key=lambda kv: (-kv[1]["pass"], kv[1]["median_tokens"] or 1e9))
        top = best[0]
        margin = top[1]["pass"] - (best[1][1]["pass"] if len(best) > 1 else 0)
        verdict[m] = {"best": top[0], "passes": f"{top[1]['pass']}/{top[1]['n']}", "margin": margin,
                      "result": "a result only if it survives a repeat" if margin <= 1 else "clear at this n"}
    curve = {m: {"passes_by_effort": {e: a["pass"] for e, a in arms.items()},
                 "thinking_by_effort": {e: {"median": a["median_thinking"], "max": a["max_thinking"]}
                                        for e, a in arms.items()},
                 "at_best": {"effort": verdict[m]["best"],
                             "max_thinking_of_passes": arms[verdict[m]["best"]]["max_thinking_of_passes"],
                             "finished_by_length": arms[verdict[m]["best"]]["length"]}}
             for m, arms in out.items()}
    return {"arms": out, "verdict": verdict, "curve": curve}


def selftest() -> int:
    bad = 0
    good = {"interval_merge": "def merge(iv):\n    iv=sorted(iv)\n    out=[]\n    for s,e in iv:\n"
                              "        if out and s<=out[-1][1]:\n            out[-1][1]=max(out[-1][1],e)\n"
                              "        else:\n            out.append([s,e])\n    return out\n"}
    t = TASKS[0]
    ok, _ = grade(extract("x\n```python\n" + good["interval_merge"] + "```\n"), t["tests"])
    bad += not ok
    print(("ok    " if ok else "FAIL  ") + "a correct answer passes its asserts")
    ok2, _ = grade(extract("```python\ndef merge(iv):\n    return iv\n```"), t["tests"])
    bad += ok2
    print(("ok    " if not ok2 else "FAIL  ") + "a wrong one fails")
    s = summarise([{"model": "bonsai", "effort": "medium", "status": 200, "pass": True, "completion_tokens": 10,
                    "s": 1}, {"model": "bonsai", "effort": "xhigh", "status": 200, "pass": False,
                              "completion_tokens": 50, "s": 3}])
    ok3 = s["verdict"]["bonsai"]["best"] == "medium" and "repeat" in s["verdict"]["bonsai"]["result"]         and "curve" in s and CAP is None
    bad += not ok3
    print(("ok    " if ok3 else "FAIL  ") + "the verdict: most passes, and a one-task margin is not a result")
    print(f"\n{'all passed' if not bad else f'{bad} FAILED'}")
    return 1 if bad else 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--key-file")
    ap.add_argument("--out")
    ap.add_argument("--models", default="mirai-s,flash-next",
                    help="bonsai's medium is bonsai-ada-surgery's measurement (the operator's tune)")
    ap.add_argument("--efforts", default=",".join(EFFORTS))
    ap.add_argument("--cap", type=int, default=CAP, help="a thinking cap: NEVER the default (operator 2026-09-29)")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if not a.key_file or not a.out:
        print("refusing: --key-file and --out (this uses the GPU: run it in a granted window)")
        return 2
    key = open(a.key_file, encoding="utf-8").read().strip()
    os.makedirs(a.out, exist_ok=True)
    rows: list[dict] = []
    path = os.path.join(a.out, "effort.jsonl")
    # model-major: one swap per model, not one per run
    for model in a.models.split(","):
        for rep in range(a.reps):
            for task in TASKS:
                for effort in a.efforts.split(","):
                    r = ask(key, task, model, effort, a.cap, SEED + rep)
                    r.update(model=model, effort=effort, task=task["id"], rep=rep)
                    rows.append(r)
                    with open(path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(r) + "\n")
                    print(json.dumps({k: r.get(k) for k in ("model", "effort", "task", "status", "pass", "finish",
                                                            "completion_tokens", "s", "served_by")}), flush=True)
    s = summarise(rows)
    s.update(n_tasks=len(TASKS), reps=a.reps, cap=a.cap, seed=SEED)
    json.dump(s, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    print(json.dumps({"verdict": s["verdict"], "curve": s["curve"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
