#!/usr/bin/env python
"""Does a second context actually buy main-context economy?

THE CLAIM UNDER TEST, in mcp/shomen.py's own words:

    "Six thousand tokens of searching becomes a two hundred token finding."

That sentence is the load-bearing justification for the whole dual-hemisphere
construct. It has never been measured. This measures it.

TWO ARMS, ONE DIFFERENCE

Arm A -- MAIN-CONTEXT. The model runs the search tools itself. All the
searching, every tool result and all the scaffolding stays in the context that
also has to hold the answer.

Arm B -- SECOND-CONTEXT. A main context holding only the question delegates to
`shomen.investigate()`. The searching happens over there; only the finding
crosses back.

WHY ARM A IS `shomen.investigate()` ITSELF

The obvious way to build arm A is to write a second tool loop with a
main-assistant system prompt. That introduces a prompt difference, a loop
difference and a stopping-rule difference on top of the one difference being
measured, and any of them could produce the result.

So arm A calls `shomen.investigate()` -- the identical loop, the identical
system prompt, the identical tools, the identical hop budget -- and counts its
token spend as MAIN-context tokens. That is precisely the counterfactual the
docstring names: "Done in the user's own context, all of it stays there."

Arm B runs the same `investigate()` and adds a main context on top of it. The
arms are therefore the same work, differing only in who pays for it. Nothing
else varies, so a difference is attributable.

Arm B's main model is offered ONLY `delegate_investigation`. If it also had
the search tools it could decline to delegate, and the arm would silently
become arm A on some fraction of rows -- self-selection destroying the
comparison. Forcing the channel is a stated design choice, not an oversight:
this measures what delegation costs and buys, not how often a model chooses it.

WHAT IS COUNTED, AND ON WHICH AXIS (PROTOCOL rule 13)

Two different token numbers get conflated constantly, so both are reported:

  tokens_cum  Sum of `usage.total_tokens` over every hop. In a re-prefilled
              tool loop hop k re-pays for the whole prefix, so this is what
              the work COSTS. It is the figure `shomen.helper_tokens`
              reports and the one the tool advertises to the caller.

  ctx_peak    max(prompt_tokens + completion_tokens) over the hops: how much
              WINDOW the work occupies at its widest. This is the number the
              context-economy argument is actually about -- "the window whose
              recall degrades over" -- and it is not the same axis as cost.

  callosum    For arm B only, and measured rather than estimated: the growth
              in the main context's prompt_tokens between the hop that issued
              the delegation and the hop that read the finding. That is
              exactly what crossed back, counted by the server's own
              tokenizer. "Two hundred token finding" is a claim about this.

`shomen.investigate()` returns only a summed `helper_tokens`, so per-hop
usage is captured by instrumenting `shomen._post` from this harness. That
touches nothing under mcp/ on disk; it is a measurement probe, and the harness
asserts that the instrumented sum equals the module's own accumulated figure,
which is also the check that the accumulation fix is really in force.

GRADING IS DETERMINISTIC (PROTOCOL rule 5)

Every task carries required file paths and required facts with explicit
accepted alternatives, hand-written against the indexed source. A row is
correct when it cites a required path AND states every required fact. No model
judges anything: grading is a pure function of the stored answer text, so it
re-runs over an existing results file without spending a token.

CONTAMINATION (PROTOCOL rule 11)

The live index is three.js, which is in every training set. Half the task set
is therefore deliberately built from `@deprecated rNNN` markers in a `187dev`
working tree -- facts that are local, exact, and recent enough to be poor
candidates for recall. The quality comparison is still labelled contaminated.

RESUMABLE

Rows append as they finish. A row carrying `error` is NOT a completion and is
retried on the next run, because a run that records its own failures as done
is how a previous benchmark got poisoned.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import statistics
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "mcp"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

# Never write to the real corpus. Two runs polluted it; this is set BEFORE
# code_search is imported, because corpus.py reads the variable at import.
os.environ.setdefault(
    "YAMADORI_CORPUS_DB",
    os.path.join(os.environ.get("TEMP", os.path.join(ROOT, "bench")),
                 "context_economy_corpus.sqlite3"))

INDEX_DB = os.environ.get("CONTEXT_ECONOMY_INDEX",
                          os.path.join(ROOT, "index", "code.sqlite3"))
os.environ["CODE_INDEX_DB"] = INDEX_DB

import code_search as cs        # noqa: E402
import shomen               # noqa: E402
from livecodebench import api_key, mcnemar, wilson   # noqa: E402

TASKS = os.path.join(HERE, "context_economy_tasks.jsonl")
OUT = os.path.join(HERE, "context_economy_results.jsonl")

ARMS = ("main_context", "second_context")

# The search surface both arms get. summarize_text is excluded because it
# calls the model, which would add generation this harness does not attribute
# to either context; run_check is excluded because it shells out to npm in a
# project root and answers no question in the task set. Everything the tasks
# need -- lexical search, symbol definition, references, file reads -- is here.
TOOL_NAMES = ("find_by_meaning", "find_by_pattern", "find_definition_opt",
              "find_references", "read_file_range", "describe_index")


def search_tools() -> list[dict]:
    return [{"type": "function",
             "function": {"name": t["name"], "description": t["description"],
                          "parameters": t["inputSchema"]}}
            for t in cs.TOOLS if t["name"] in TOOL_NAMES]


def run_tool(name: str, args: dict) -> str:
    """Dispatch straight into code_search, with no proxy in the path.

    The proxy adds a preamble, a tier gate and a tool merge, none of which is
    under test and all of which would land on both arms unevenly -- arm B's
    investigator goes through `sub_run`, arm A's model would go through the
    request path. Calling the same function for both removes that.
    """
    prev = cs.INDEX_DB
    cs.INDEX_DB = INDEX_DB
    os.environ["CODE_INDEX_DB"] = INDEX_DB
    try:
        resp = cs.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                          "params": {"name": name, "arguments": args}})
        if resp is None or "result" not in resp:
            return f"{name} failed: {json.dumps(resp)[:400]}"
        return resp["result"]["content"][0]["text"]
    except Exception as e:                                       # noqa: BLE001
        return f"{name} failed: {type(e).__name__}: {e}"
    finally:
        cs.INDEX_DB = prev


# ---------------------------------------------------------------------------
# The measurement probe.
# ---------------------------------------------------------------------------
_HOPS: list[dict] = []
_REAL_POST = shomen._post


def _instrumented_post(path: str, payload: dict, timeout: int = 1800) -> dict:
    # 1800, the default of the shomen._post it replaces. At 900 this shim
    # silently halved production's timeout, so the benchmark measured a
    # stricter stack than the one it claims to (docs/CONSTRAINTS.md 13).
    t0 = time.time()
    d = _REAL_POST(path, payload, timeout)
    u = d.get("usage") or {}
    _HOPS.append({"prompt": int(u.get("prompt_tokens") or 0),
                  "completion": int(u.get("completion_tokens") or 0),
                  "total": int(u.get("total_tokens") or 0),
                  "ms": round((time.time() - t0) * 1000)})
    return d


shomen._post = _instrumented_post


def _reset() -> None:
    _HOPS.clear()


def _roll(hops: list[dict]) -> dict:
    if not hops:
        return {"tokens_cum": 0, "ctx_peak": 0, "hops": 0,
                "prompt_first": 0, "prompt_last": 0}
    return {"tokens_cum": sum(h["total"] for h in hops),
            "ctx_peak": max(h["prompt"] + h["completion"] for h in hops),
            "hops": len(hops),
            "prompt_first": hops[0]["prompt"],
            "prompt_last": hops[-1]["prompt"]}


# ---------------------------------------------------------------------------
# Arm A -- the searching happens in the context that must also hold the answer.
# ---------------------------------------------------------------------------
def run_main_context(task: dict, hops: int) -> dict:
    _reset()
    t0 = time.time()
    res = shomen.investigate(task["question"], search_tools(), run_tool,
                                 hops=hops)
    seen = list(_HOPS)
    roll = _roll(seen)
    # PROTOCOL rule 3: check the harness before believing it. If the module's
    # own accumulated figure and the per-hop probe disagree, the accumulation
    # is not doing what it is documented to do and no token table below means
    # anything.
    drift = roll["tokens_cum"] - int(res.get("helper_tokens") or 0)
    answer = res.get("finding") or ""
    return {
        "arm": "main_context",
        "ok": bool(res.get("ok")),
        "answer": answer,
        "tool_calls": res.get("hops"),
        "tools_used": res.get("tools_used") or [],
        "cited_by_module": res.get("cited") or [],
        "unsupported_citations": res.get("unsupported") or [],
        "main_tokens_cum": roll["tokens_cum"],
        "main_ctx_peak": roll["ctx_peak"],
        "main_hops": roll["hops"],
        "second_tokens_cum": 0,
        "second_ctx_peak": 0,
        "second_hops": 0,
        "callosum_tokens": None,
        "total_tokens": roll["tokens_cum"],
        "seconds": round(time.time() - t0, 1),
        "usage_drift": drift,
        "hop_usage": seen,
        "error": None if res.get("ok") else str(res.get("finding"))[:300],
    }


# ---------------------------------------------------------------------------
# Arm B -- a main context that delegates and then writes from the finding.
# ---------------------------------------------------------------------------
# Deliberately the same shape as the hemisphere's own system prompt: a stated
# budget, an ordering, a length cap and a citation requirement. The only thing
# that differs is the means. Instructing one arm to be brief and the other to
# explore would produce a token difference that is the instruction, not the
# architecture (PROTOCOL rule 15).
MAIN_SYSTEM = """You are answering a question about a codebase for another
engineer. You cannot search yourself. You have one tool: delegate_investigation,
which hands a self-contained question to a second model that has the search
tools and its own separate context, and returns a short finding with file
citations.

BUDGET: at most {hops} turns. Plan inside it. Delegate once with a question
that names what you need, read the finding, then write the answer.

Work in this order:
1. Delegate the investigation.
2. Stop as soon as you can answer. Do not delegate again out of interest.
3. Write the answer.

The answer must be under 200 words and must cite the file paths and line
numbers the finding gives you. Write what the code DOES, not what you did to
find it. If the finding does not answer the question, say exactly that -- an
honest "the index does not contain this" is useful and a guess is not.

Never state anything the investigation did not report back."""


def run_second_context(task: dict, hops: int, outer_hops: int) -> dict:
    t0 = time.time()
    outer: list[dict] = []
    inner_rolls: list[dict] = []
    tool_calls_inner = 0
    tools_used: set = set()
    cited: list = []
    unsupported: list = []
    delegations = 0
    convo = [{"role": "system", "content": MAIN_SYSTEM.format(hops=outer_hops)},
             {"role": "user", "content": task["question"]}]
    answer = ""
    err = None
    crossings: list[int] = []
    # The delegated question is NOT the task question -- the main model
    # rewrites it -- and the rewrite is a candidate explanation for the
    # investigator doing more work than the identical loop does in arm A.
    # Unrecorded, that stays a correlation (PROTOCOL rule 13).
    asked: list[str] = []
    prompt_at_delegation = None

    for hop in range(outer_hops):
        last = hop == outer_hops - 1
        if last:
            convo.append({"role": "user", "content":
                          "Stop delegating. Write the answer now from what you "
                          "have, and cite the paths you were given. If it is "
                          "incomplete, say what is missing."})
        payload = {"model": shomen.MODEL, "messages": convo,
                   "tools": [] if last else [shomen.TOOL],
                   "max_tokens": 2500, "temperature": 0.2}
        try:
            d = _REAL_POST("/v1/chat/completions", payload)
        except Exception as e:                                   # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
            break
        u = d.get("usage") or {}
        outer.append({"prompt": int(u.get("prompt_tokens") or 0),
                      "completion": int(u.get("completion_tokens") or 0),
                      "total": int(u.get("total_tokens") or 0)})
        if prompt_at_delegation is not None:
            # Measured, not estimated: the whole growth of the main context's
            # prompt between issuing the delegation and reading the result IS
            # what crossed the callosum, counted by the server's tokenizer.
            crossings.append(outer[-1]["prompt"] - prompt_at_delegation)
            prompt_at_delegation = None

        msg = d["choices"][0]["message"]
        calls = msg.get("tool_calls") or []
        if not calls:
            answer = (msg.get("content") or "").strip()
            if not answer and msg.get("reasoning_content"):
                err = "main context produced reasoning but no answer"
            break

        convo.append({"role": "assistant", "content": msg.get("content") or "",
                      "tool_calls": calls})
        for c in calls:
            fn = c.get("function", {}).get("name", "")
            try:
                args = json.loads(c["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            if fn != "delegate_investigation":
                convo.append({"role": "tool", "tool_call_id": c.get("id"),
                              "content": f"unknown tool {fn}"})
                continue
            q = args.get("question")
            if not isinstance(q, str) or len(q.strip()) < 8:
                # The proxy's argument gate, reproduced so a malformed call
                # costs what it costs there rather than starting a GPU run.
                convo.append({"role": "tool", "tool_call_id": c.get("id"),
                              "content": ("`question` must be a non-empty "
                                          "string of at least 8 characters. "
                                          "Nothing was run.")})
                continue
            asked.append(q)
            _reset()
            res = shomen.investigate(q, search_tools(), run_tool,
                                         context=args.get("context", ""),
                                         hops=hops)
            roll = _roll(list(_HOPS))
            roll["drift"] = roll["tokens_cum"] - int(res.get("helper_tokens") or 0)
            inner_rolls.append(roll)
            delegations += 1
            tool_calls_inner += int(res.get("hops") or 0)
            tools_used |= set(res.get("tools_used") or [])
            cited += res.get("cited") or []
            unsupported += res.get("unsupported") or []
            # Byte-for-byte what mcp/proxy.py hands back, so the main context
            # pays exactly what it pays in production.
            back = (res["finding"] + "\n\n"
                    + f"[investigated in a separate context: {res['hops']} tool "
                      f"calls, {res['helper_tokens']} tokens spent there, "
                      f"{res['seconds']}s. None of that entered this "
                      f"conversation. Trace handle {res['handle']}.]")
            convo.append({"role": "tool", "tool_call_id": c.get("id"),
                          "content": back})
            prompt_at_delegation = outer[-1]["prompt"]

    o = _roll(outer)
    inner_cum = sum(r["tokens_cum"] for r in inner_rolls)
    inner_peak = max([r["ctx_peak"] for r in inner_rolls] or [0])
    if not answer and not err:
        err = f"no answer after {outer_hops} main-context turns"
    return {
        "arm": "second_context",
        "ok": err is None,
        "answer": answer,
        "tool_calls": tool_calls_inner,
        "tools_used": sorted(tools_used),
        "cited_by_module": sorted(set(cited)),
        "unsupported_citations": sorted(set(unsupported)),
        "main_tokens_cum": o["tokens_cum"],
        "main_ctx_peak": o["ctx_peak"],
        "main_hops": o["hops"],
        "second_tokens_cum": inner_cum,
        "second_ctx_peak": inner_peak,
        "second_hops": sum(r["hops"] for r in inner_rolls),
        "delegations": delegations,
        "callosum_tokens": sum(crossings) if crossings else None,
        "callosum_each": crossings,
        "delegated_questions": asked,
        "total_tokens": o["tokens_cum"] + inner_cum,
        "seconds": round(time.time() - t0, 1),
        "usage_drift": sum(r["drift"] for r in inner_rolls),
        "hop_usage": outer,
        "error": err,
    }


# ---------------------------------------------------------------------------
# Grading. Deterministic, and a pure function of the stored answer.
# ---------------------------------------------------------------------------
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").replace("\\", "/")).lower()


def grade(task: dict, answer: str) -> dict:
    a = _norm(answer)
    paths = [p.lower() for p in task["paths"]]
    full = [p for p in paths if p in a]
    base = [p for p in paths if os.path.basename(p) in a]
    missing = []
    for g in task["facts"]:
        if not any(_norm(alt) in a for alt in g):
            missing.append(g[0])
    cite_ok = bool(full or base)
    fact_ok = not missing
    return {"cite_ok": cite_ok, "cite_exact": bool(full),
            "fact_ok": fact_ok, "missing_facts": missing,
            "correct": cite_ok and fact_ok}


# ---------------------------------------------------------------------------
# Preflight. PROTOCOL rule 1: prove every component alive, individually, with
# a call that would fail loudly if it were not.
# ---------------------------------------------------------------------------
def preflight(check_proxy: bool = False) -> bool:
    ok = True
    print("PREFLIGHT")
    try:
        with urllib.request.urlopen(
                f"{shomen.UPSTREAM}/v1/models", timeout=20) as r:
            ids = [m["id"] for m in json.load(r).get("data", [])]
        have = shomen.MODEL in ids
        print(f"  upstream {shomen.UPSTREAM}: {ids}")
        print(f"  model '{shomen.MODEL}' present: {have}")
        ok &= have
    except Exception as e:                                       # noqa: BLE001
        print(f"  upstream UNREACHABLE: {type(e).__name__}: {e}")
        ok = False

    try:
        db = sqlite3.connect(INDEX_DB)
        n = {t: db.execute(f"select count(*) from {t}").fetchone()[0]
             for t in ("chunks", "defs", "refs")}
        roots = [r[0] for r in db.execute("select * from roots")]
        print(f"  index {os.path.basename(INDEX_DB)}: {n}  roots={roots}")
        ok &= n["chunks"] > 0 and n["defs"] > 0
    except Exception as e:                                       # noqa: BLE001
        print(f"  index UNREADABLE: {type(e).__name__}: {e}")
        ok = False

    # Not "it returned 200" -- it has to do the actual job and return something
    # only a live index could produce.
    probes = [("find_definition_opt", {"symbol": "WebGPUBackend"}, "WebGPUBackend.js"),
              ("find_by_pattern", {"pattern": "REVISION", "max_results": 3}, "constants.js"),
              ("read_file_range", {"path": "constants.js", "start": 1, "end": 2}, "REVISION"),
              ("find_references", {"symbol": "Vector3"}, ".js"),
              ("find_by_meaning", {"query": "how are shadows rendered", "top_k": 2}, "."),
              ("describe_index", {}, "chunks")]
    for name, args, want in probes:
        out = run_tool(name, args)
        good = want.lower() in (out or "").lower() and len(out or "") > 20
        print(f"  tool {name:<20} {len(out or ''):>6} chars  alive={good}")
        ok &= good

    print(f"  semantic retriever: CODE_SEARCH_SEMANTIC="
          f"{os.environ.get('CODE_SEARCH_SEMANTIC', '0')} "
          f"(0 = lexical only, no embedding server touched)")
    print(f"  corpus db (must NOT be index/corpus.sqlite3): "
          f"{os.environ['YAMADORI_CORPUS_DB']}")

    if check_proxy:
        key = api_key()
        req = urllib.request.Request("http://127.0.0.1:1234/v1/models")
        if key:
            req.add_header("Authorization", f"Bearer {key}")
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                print(f"  proxy :1234 (key {'found' if key else 'MISSING'}): "
                      f"{[m['id'] for m in json.load(r).get('data', [])]}")
        except Exception as e:                                   # noqa: BLE001
            print(f"  proxy :1234: {type(e).__name__}: {e}")

    print(f"  PREFLIGHT {'PASS' if ok else 'FAIL'}\n")
    return ok


# ---------------------------------------------------------------------------
def load_tasks(path: str) -> list[dict]:
    out = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def load_done(path: str) -> set:
    """Completed pairs only. An ERROR is not a completion, and is retried."""
    done = set()
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            try:
                r = json.loads(line)
            except Exception:                                    # noqa: BLE001
                continue
            if r.get("error"):
                continue
            done.add((r["id"], r["arm"]))
    return done


def load_rows(path: str) -> list[dict]:
    rows = []
    if not os.path.exists(path):
        return rows
    for line in open(path, encoding="utf-8"):
        try:
            rows.append(json.loads(line))
        except Exception:                                        # noqa: BLE001
            continue
    return rows


def sign_test(wins: int, losses: int) -> float:
    """Two-sided exact sign test -- the same arithmetic as McNemar."""
    return mcnemar(wins, losses)


def _med(xs):
    return statistics.median(xs) if xs else 0


def report(path: str, tasks_path: str = TASKS) -> None:
    rows = load_rows(path)
    tasks = {t["id"]: t for t in load_tasks(tasks_path)}
    if not rows:
        print("no results yet")
        return

    # Latest non-error row per (id, arm) wins; an error row is kept only if
    # nothing better exists, so failures stay visible without displacing a
    # later success.
    best: dict = {}
    errs: list = []
    for r in rows:
        k = (r["id"], r["arm"])
        if r.get("error"):
            errs.append(r)
            best.setdefault(k, r)
        else:
            best[k] = r
    for (i, arm), r in best.items():
        if not r.get("error"):
            r["grade"] = grade(tasks[i], r.get("answer") or "")

    paired = [i for i in tasks
              if all((i, a) in best and not best[(i, a)].get("error")
                     for a in ARMS)]
    n = len(paired)

    print("=" * 78)
    print("CONTEXT ECONOMY -- does a second context buy main-context economy?")
    print("=" * 78)
    print(f"tasks in set: {len(tasks)}   complete pairs: {n}   "
          f"rows on file: {len(rows)}   error rows: {len(errs)}")
    print(f"index: {INDEX_DB}")
    print(f"model: {shomen.MODEL} @ {shomen.UPSTREAM}   "
          f"grading: deterministic (paths + required facts)")

    if errs:
        print("\nERRORS -- a distinct outcome, not a wrong answer (rule 3)")
        kinds: dict = {}
        for r in errs:
            kinds[(r["arm"], str(r["error"])[:60])] = \
                kinds.get((r["arm"], str(r["error"])[:60]), 0) + 1
        for (arm, k), c in sorted(kinds.items(), key=lambda kv: -kv[1]):
            print(f"  {arm:<15} {c:>3}  {k}")

    drift = [r.get("usage_drift", 0) for r in best.values() if not r.get("error")]
    print(f"\nHARNESS CHECK  usage accumulation drift "
          f"(probe sum minus module's own figure): "
          f"min {min(drift or [0])}, max {max(drift or [0])} "
          f"-- nonzero means the accumulated usage figure is not what it says")

    if n == 0:
        print("\nno complete pairs yet")
        return

    # ---- power, stated before the numbers (rule 4) --------------------------
    print(f"\nPOWER.  n={n} paired rows. McNemar's exact test on b+c discordant "
          f"pairs\nreaches p<0.05 only at b+c>=6 with a 6-0 split. At this n the "
          f"quality\ncomparison can detect a LARGE effect and nothing smaller; "
          f"a null here means\n'this experiment could not have found a small "
          f"difference', not 'no difference'.\nThe token comparison is a "
          f"different matter -- it is near-deterministic per row\nand does not "
          f"need n to be readable.")

    def col(arm, key):
        return [best[(i, arm)][key] for i in paired
                if best[(i, arm)].get(key) is not None]

    # ---- 1. quality --------------------------------------------------------
    print("\n" + "-" * 78)
    print("1. ANSWER QUALITY  (deterministic: cites a required path AND states "
          "every fact)")
    print("-" * 78)
    print(f"{'arm':<16}{'correct':>9}{'rate':>8}{'95% CI':>18}"
          f"{'cited ok':>10}{'facts ok':>10}")
    for arm in ARMS:
        g = [best[(i, arm)]["grade"] for i in paired]
        k = sum(x["correct"] for x in g)
        lo, hi = wilson(k, n)
        print(f"{arm:<16}{k:>6}/{n:<3}{k / n:>7.1%}  [{lo:>5.1%},{hi:>6.1%}]"
              f"{sum(x['cite_ok'] for x in g):>10}"
              f"{sum(x['fact_ok'] for x in g):>10}")

    b = c = both = neither = 0
    disc = []
    for i in paired:
        a_ok = best[(i, "main_context")]["grade"]["correct"]
        s_ok = best[(i, "second_context")]["grade"]["correct"]
        if a_ok and s_ok:
            both += 1
        elif a_ok and not s_ok:
            b += 1
            disc.append((i, "main_context"))
        elif s_ok and not a_ok:
            c += 1
            disc.append((i, "second_context"))
        else:
            neither += 1
    p = mcnemar(b, c)
    print(f"\nPAIRED.  both correct {both}   both wrong {neither}   "
          f"discordant {b + c}")
    print(f"  main-context only correct   b = {b}")
    print(f"  second-context only correct c = {c}")
    print(f"  McNemar exact p = {p:.4f}"
          + ("   (no discordant pairs: the arms are indistinguishable on "
             "quality here)" if b + c == 0 else ""))
    if disc:
        print("\n  THE DISCORDANT PAIRS -- the entire quality signal:")
        for i, winner in disc:
            loser = ARMS[0] if winner == ARMS[1] else ARMS[1]
            g = best[(i, loser)]["grade"]
            why = ("no required path cited" if not g["cite_ok"]
                   else "missing facts: " + ", ".join(g["missing_facts"]))
            print(f"    {i} [{tasks[i]['kind']}]  {winner} won; "
                  f"{loser} failed -- {why}")

    # ---- 2. the primary outcome -------------------------------------------
    print("\n" + "-" * 78)
    print("2. MAIN-CONTEXT TOKENS  <-- THIS IS THE CLAIM")
    print("-" * 78)
    a_cum, s_cum = col("main_context", "main_tokens_cum"), col("second_context", "main_tokens_cum")
    a_pk, s_pk = col("main_context", "main_ctx_peak"), col("second_context", "main_ctx_peak")
    print(f"{'measure':<34}{'arm A main-ctx':>16}{'arm B main-ctx':>16}"
          f"{'B/A':>9}")
    for label, aa, ss in (("cumulative tokens (median)", _med(a_cum), _med(s_cum)),
                          ("cumulative tokens (mean)",
                           round(statistics.mean(a_cum)) if a_cum else 0,
                           round(statistics.mean(s_cum)) if s_cum else 0),
                          ("peak window occupied (median)", _med(a_pk), _med(s_pk)),
                          ("peak window occupied (max)",
                           max(a_pk or [0]), max(s_pk or [0]))):
        ratio = (ss / aa) if aa else float("nan")
        print(f"{label:<34}{aa:>16,}{ss:>16,}{ratio:>9.2f}")

    wins = sum(1 for i in paired
               if best[(i, "second_context")]["main_tokens_cum"]
               < best[(i, "main_context")]["main_tokens_cum"])
    losses = sum(1 for i in paired
                 if best[(i, "second_context")]["main_tokens_cum"]
                 > best[(i, "main_context")]["main_tokens_cum"])
    print(f"\nPAIRED, per task: second context used FEWER main-context tokens "
          f"on {wins}/{n},\n  more on {losses}/{n}, "
          f"exact sign test p = {sign_test(wins, losses):.4g}")
    saved = [best[(i, "main_context")]["main_tokens_cum"]
             - best[(i, "second_context")]["main_tokens_cum"] for i in paired]
    print(f"  per-task main-context saving: median {_med(saved):,.0f}, "
          f"min {min(saved):,}, max {max(saved):,}")
    savedp = [best[(i, "main_context")]["main_ctx_peak"]
              - best[(i, "second_context")]["main_ctx_peak"] for i in paired]
    print(f"  per-task peak-window saving:  median {_med(savedp):,.0f}, "
          f"min {min(savedp):,}, max {max(savedp):,}")

    # ---- 3. the price ------------------------------------------------------
    print("\n" + "-" * 78)
    print("3. TOTAL TOKENS ACROSS BOTH CONTEXTS  <-- what the saving costs")
    print("-" * 78)
    a_tot, s_tot = col("main_context", "total_tokens"), col("second_context", "total_tokens")
    print(f"{'arm':<16}{'main ctx':>12}{'second ctx':>12}{'TOTAL':>12}"
          f"{'median total':>14}")
    print(f"{'main_context':<16}{sum(a_cum):>12,}{0:>12,}{sum(a_tot):>12,}"
          f"{_med(a_tot):>14,.0f}")
    s_inner = col("second_context", "second_tokens_cum")
    print(f"{'second_context':<16}{sum(s_cum):>12,}{sum(s_inner):>12,}"
          f"{sum(s_tot):>12,}{_med(s_tot):>14,.0f}")
    if sum(a_tot):
        print(f"\n  total-token price of delegation: "
              f"{sum(s_tot) / sum(a_tot):.2f}x arm A")
    twins = sum(1 for i in paired
                if best[(i, "second_context")]["total_tokens"]
                < best[(i, "main_context")]["total_tokens"])
    tloss = n - twins
    print(f"  paired: arm B cheaper in TOTAL on {twins}/{n}, "
          f"dearer on {tloss}/{n}, sign test p = {sign_test(twins, tloss):.4g}")

    # ---- 4. the callosum ---------------------------------------------------
    print("\n" + "-" * 78)
    print('4. "SIX THOUSAND TOKENS OF SEARCHING BECOMES A TWO HUNDRED TOKEN '
          'FINDING"')
    print("-" * 78)
    cal = col("second_context", "callosum_tokens")
    inner = [best[(i, "second_context")]["second_tokens_cum"] for i in paired]
    inner_pk = [best[(i, "second_context")]["second_ctx_peak"] for i in paired]
    print(f"  investigation spent, per task (median)      "
          f"{_med(inner):>10,.0f} tokens")
    print(f"  investigation peak window (median)          "
          f"{_med(inner_pk):>10,.0f} tokens")
    if cal:
        print(f"  what actually crossed back (median)         "
              f"{_med(cal):>10,.0f} tokens   n={len(cal)}")
        print(f"  crossing-back range                         "
              f"{min(cal):,} to {max(cal):,} tokens")
        if _med(cal):
            print(f"  compression achieved across the callosum    "
                  f"{_med(inner) / _med(cal):>10.1f}x")
        print("  (measured as the growth of the main context's prompt_tokens "
              "between\n   issuing the delegation and reading the finding -- "
              "the server's own count,\n   and it includes the tool-call "
              "message and the cost note, not just prose)")
    else:
        print("  no delegation completed, so nothing crossed back")
    dele = [best[(i, "second_context")].get("delegations", 0) for i in paired]
    never = [i for i in paired
             if not best[(i, "second_context")].get("delegations")]
    if dele:
        print(f"  delegations per task: median {_med(dele):.0f}, "
              f"max {max(dele)}, tasks with >1: "
              f"{sum(1 for d in dele if d > 1)}")
    if never:
        print(f"  ARM B DID NOT DELEGATE on {len(never)}/{n} rows "
              f"({', '.join(never[:8])}).\n  Those rows answered from the "
              f"model's own knowledge with no retrieval at all, and they are\n"
              f"  arm B in name only -- they flatter its token numbers and "
              f"carry no retrieval signal.")

    # ---- 5. clock ----------------------------------------------------------
    print("\n" + "-" * 78)
    print("5. WALL CLOCK")
    print("-" * 78)
    a_s, s_s = col("main_context", "seconds"), col("second_context", "seconds")
    print(f"{'arm':<16}{'median s':>10}{'mean s':>10}{'min':>9}{'max':>9}"
          f"{'total s':>11}")
    for arm, xs in (("main_context", a_s), ("second_context", s_s)):
        print(f"{arm:<16}{_med(xs):>10.1f}"
              f"{statistics.mean(xs) if xs else 0:>10.1f}"
              f"{min(xs or [0]):>9.1f}{max(xs or [0]):>9.1f}{sum(xs):>11.0f}")
    if _med(a_s):
        print(f"\n  arm B takes {_med(s_s) / _med(a_s):.2f}x the wall clock of "
              f"arm A at the median")

    # ---- 6. behaviour ------------------------------------------------------
    print("\n" + "-" * 78)
    print("6. WHAT EACH ARM ACTUALLY DID")
    print("-" * 78)
    print(f"{'arm':<16}{'tool calls':>12}{'model hops':>12}"
          f"{'answer chars':>14}{'unsupported cites':>19}")
    for arm in ARMS:
        tc = col(arm, "tool_calls")
        hp = [best[(i, arm)]["main_hops"] + best[(i, arm)]["second_hops"]
              for i in paired]
        ch = [len(best[(i, arm)].get("answer") or "") for i in paired]
        un = sum(len(best[(i, arm)].get("unsupported_citations") or [])
                 for i in paired)
        print(f"{arm:<16}{_med(tc):>12.1f}{_med(hp):>12.1f}{_med(ch):>14.0f}"
              f"{un:>19}")

    zero = [i for i in paired if not best[(i, "main_context")]["tool_calls"]]
    if zero:
        print(f"\n  CONTAMINATION WATCH: {len(zero)}/{n} arm-A rows answered "
              f"with ZERO tool calls\n  ({', '.join(zero[:8])}). three.js is in "
              f"every training set; a row answered\n  from memory is not "
              f"measuring retrieval on either arm.")

    print("\n" + "-" * 78)
    print("7. BY TASK KIND")
    print("-" * 78)
    kinds = sorted({tasks[i]["kind"] for i in paired})
    print(f"{'kind':<12}{'n':>4}{'A correct':>11}{'B correct':>11}"
          f"{'A main tok':>12}{'B main tok':>12}")
    for k in kinds:
        ids = [i for i in paired if tasks[i]["kind"] == k]
        ak = sum(best[(i, "main_context")]["grade"]["correct"] for i in ids)
        sk = sum(best[(i, "second_context")]["grade"]["correct"] for i in ids)
        print(f"{k:<12}{len(ids):>4}{ak:>11}{sk:>11}"
              f"{_med([best[(i, 'main_context')]['main_tokens_cum'] for i in ids]):>12,.0f}"
              f"{_med([best[(i, 'second_context')]['main_tokens_cum'] for i in ids]):>12,.0f}")
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks", default=TASKS)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", default="", help="comma-separated task ids")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--hops", type=int, default=None,
                    help="searching turns, identical for both arms (default: "
                         "none -- ends when the investigation stops or its "
                         "context share is full)")
    ap.add_argument("--outer-hops", type=int, default=4,
                    help="turns arm B's MAIN context gets")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--no-preflight", action="store_true")
    ap.add_argument("--proxy-check", action="store_true")
    a = ap.parse_args()

    if a.report_only:
        report(a.out, a.tasks)
        return

    if not a.no_preflight and not preflight(a.proxy_check):
        print("preflight failed -- not running. Rule 1: bring the whole "
              "system up before judging any part of it.")
        sys.exit(1)

    tasks = load_tasks(a.tasks)
    if a.only:
        want = {x.strip() for x in a.only.split(",")}
        tasks = [t for t in tasks if t["id"] in want]
    if a.limit:
        tasks = tasks[:a.limit]
    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    done = load_done(a.out)

    todo = [(t, arm) for t in tasks for arm in arms
            if (t["id"], arm) not in done]
    print(f"{len(tasks)} tasks x {len(arms)} arms; {len(todo)} to run "
          f"({len(done)} already complete)\n")

    # SEQUENTIAL. The hemisphere is a singleton behind a semaphore and takes
    # the helper lane; two arms in flight would queue on each other and the
    # wall clock would measure the queue. Both arms of a task run back to
    # back, so any drift in the server lands on both.
    t_start = time.time()
    for k, (task, arm) in enumerate(todo, 1):
        print(f"[{k}/{len(todo)}] {task['id']} {arm} ... ", end="", flush=True)
        t0 = time.time()
        try:
            if arm == "main_context":
                row = run_main_context(task, a.hops)
            else:
                row = run_second_context(task, a.hops, a.outer_hops)
        except KeyboardInterrupt:
            print("\ninterrupted -- results so far are on file and the run "
                  "resumes where it stopped")
            break
        except Exception as e:                                   # noqa: BLE001
            row = {"arm": arm, "ok": False, "answer": "", "error":
                   f"{type(e).__name__}: {e}", "seconds":
                   round(time.time() - t0, 1), "main_tokens_cum": 0,
                   "main_ctx_peak": 0, "main_hops": 0, "second_tokens_cum": 0,
                   "second_ctx_peak": 0, "second_hops": 0,
                   "total_tokens": 0, "callosum_tokens": None,
                   "tool_calls": 0, "tools_used": [], "usage_drift": 0,
                   "cited_by_module": [], "unsupported_citations": []}
        row["id"] = task["id"]
        row["kind"] = task["kind"]
        row["question"] = task["question"]
        row["hops_budget"] = a.hops
        row["model"] = shomen.MODEL
        row["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        g = grade(task, row.get("answer") or "") if not row.get("error") else None
        row["grade"] = g
        # Checkpoint AS WE GO. A long run must survive interruption.
        with open(a.out, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
        mark = "ERR " + str(row["error"])[:60] if row.get("error") else \
            ("correct" if g and g["correct"] else "wrong  ")
        print(f"{row['seconds']:>6.1f}s  main {row['main_tokens_cum']:>7,}  "
              f"second {row['second_tokens_cum']:>7,}  {mark}")

    print(f"\nrun took {time.time() - t_start:,.0f}s total\n")
    report(a.out, a.tasks)


if __name__ == "__main__":
    main()
