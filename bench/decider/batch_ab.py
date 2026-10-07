#!/usr/bin/env python
"""JJAVA BATCHED READS, SEQUENTIAL vs BATCHED (a GPU measurement; NOT RUN).

    python bench/decider/batch_ab.py                    # the plan, no network
    python bench/decider/batch_ab.py --plan-json        # the plan as JSON
    python bench/decider/batch_ab.py --run --gpu-go     # GPU: runs it

WHAT IT ANSWERS (docs/DECIDE-BATCH.md section 8; the engine patch
engines/patches/llama-bonsai2-ada/0042, mcp/decider_batch.py behind
YAMADORI_DECIDER_BATCH):

  1. SPEED: wall time per call and per question, the per-read path against
     ONE /decide-batch request, by questions per call (K).
  2. IDENTITY: questions never see each other, so the answers must be the
     per-read path's, up to the engine's batch noise. Per item: whether the
     ARGMAX differs, the largest |p| difference over every label (the batch-
     invariance floor, compared with decider_bonsai.TIE_BAND = 0.0034, which
     is the floor measured for prefix-cached vs cold reads), and the TIE
     FLIPS (the tie flag differing between the arms).

ITEM SETS (all asked of the same reader; every item runs the sequential arm
and the batched arm, the arm that goes first alternating across items so
drift cancels):

  jevbench   (a) JevBench's 231 public items as bench/decider/jevbench_run.py
             loads them: one question each (K = 1).
  k<K>       (b) K-question calls, K in 1, 2, 5, 10, 26: item i's state with
             its own question plus the questions of the next K-1 items (wrapping
             round the pool; semantically odd, valid for speed and identity).
             A call whose state plus questions exceed the 64k-token request
             limit (jev_api.REQUEST_LIMIT_TOKENS) is left out and counted.
  skill      (c) the labelled skill cases of the 2026-10-06 window (docs/JJAVA.md
             9; bench/skills/run_window_20261006.ps1, inject_decide.py): per case
             the stage-2 questions of variant a over the case's state as ONE
             decide_turn.Turn.decide call (a Turn per arm), then the stage-3
             question as a second call on the same Turn (the resident prefix),
             sequential vs batched.

THE READER: layout v3's jjava reader, `bonsai-a4000` (decider_target: the
llama-swap passthrough, which must already list it in GET /running; or a bare
server with --base-url). Nothing here loads a model.

GATES: the default is a DRY RUN that builds the plan and prints it with the
exact request counts and the arithmetic time estimate (the brief's fit,
~0.26 s of fixed overhead per read + ~2.3 ms per processed token: ARITHMETIC,
not a measurement). `--run` is REFUSED unless `--gpu-go` is also given (the
operator says "GPU go"), and then it runs the preflight (one GPU consumer on
the target), one capability probe (a batched call that falls back stops the
run: there would be nothing to compare), and writes
bench/decider/results/batch_ab/<run>/{items.jsonl, summary.json}: ids,
timings and distributions, never item text.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "mcp"))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "bench", "skills"))

RUN_VERSION = "batch-ab/1"
OUT_DIR = os.path.join(HERE, "results", "batch_ab")
DEFAULT_MODEL = "bonsai-a4000"      # layout v3's jjava reader (AGENTS.md)
KS = (1, 2, 5, 10, 26)              # the brief's K list
SKILL_VARIANT = "a"                 # the window's served stage-2 variant
DEFAULT_CASES = "C:/Users/jwals/octo/codex-label-kit/cases.jsonl"
# The brief's fit of the per-read path (2026-10-07): ARITHMETIC for the
# plan's estimate only.
FIXED_S_PER_READ = 0.26
S_PER_TOKEN = 0.0023
CHECK_EVERY = 25                    # preflight cadence, as jevbench_run


# ------------------------------------------------------------- the plan -----
def est_tokens(text: str) -> int:
    """chars / 3, the repo's offline stand-in for a token count
    (decide_turn._count); the run counts with the server's /tokenize."""
    return max(1, len(text) // 3)


def rename(q: dict, name: str) -> dict:
    return dict(q, name=name)


def q_tokens(q: dict, count=est_tokens) -> int:
    """What a question puts after the state, per read (its longest order)."""
    import decider_bonsai as D
    return max(count(D.question_text(c)) for c in D.rendered_orders(q))


def fits(state_tokens: int, questions: list[dict], count=est_tokens) -> bool:
    """The request limit (jev_api.REQUEST_LIMIT_TOKENS, Jev's 64k per
    request): state plus every question."""
    import jev_api as J
    return state_tokens + sum(q_tokens(q, count) for q in questions) \
        <= J.REQUEST_LIMIT_TOKENS


def k_calls(pool: list[dict], k: int, count=est_tokens) -> tuple[list, int]:
    """(calls, skipped) for K questions a call over `pool` ([{id, state,
    question}]): item i's state with its own question plus the next K-1
    items' (wrapping), each renamed q0..q<K-1>. Calls over the request limit
    are skipped."""
    calls, skipped = [], 0
    n = len(pool)
    for i, it in enumerate(pool):
        qs = [rename(pool[(i + j) % n]["question"], f"q{j}")
              for j in range(min(k, n))]
        if not fits(count(it["state"]), qs, count):
            skipped += 1
            continue
        calls.append({"set": f"k{k}", "item": it["id"], "k": len(qs),
                      "mode": "decide", "state": it["state"],
                      "questions": qs})
    return calls, skipped


def stage3_question(variant: str, items: list[dict]) -> dict:
    """inject_decide's stage 3 for a case whose stage 2 passed nothing (its
    `forced` rule): the injector's noul over the first two facts."""
    import decider_bonsai as D
    import skill_inject as I
    listed_items = [{"key": it["key"], "form": "DO", "situation": "",
                     "text": it["fact"]} for it in items[:2]]
    q, _listed = I.inject_question(listed_items)
    return D.q_noul(f"{I.QSET_INJECT}{'' if variant == 'a' else '_' + variant}",
                    q["text"])


def skill_calls(cases: list[dict], variants=(SKILL_VARIANT,),
                questions_fn=None, stage3_fn=None) -> list[dict]:
    """The window's decide_turn question sets: per case and variant, the
    stage-2 questions over the case's state as one Turn.decide, then the
    stage-3 question (inject_decide's `forced` form, over the case's first
    two facts) as a second decide on the same Turn."""
    if questions_fn is None:
        import inject_decide as ID
        questions_fn = ID.questions
    stage3_fn = stage3_fn or stage3_question
    out = []
    for c in cases:
        for v in variants:
            qs = questions_fn(v, c["items"])
            if not qs:
                continue
            out.append({"set": "skill", "item": f"{c['case']}:{v}",
                        "k": len(qs), "mode": "turn", "state": c["state"],
                        "questions": qs,
                        "followup": [stage3_fn(v, c["items"])],
                        "state_info": dict(c.get("state_info") or {},
                                           kind=c.get("kind") or "step")})
    return out


def jevbench_pool(clone: str) -> list[dict]:
    """JevBench's public items as a pool: [{id, state, question, tier}]
    (bench/decider/jevbench_run.py's loader and mapping, reused). Raises
    when the clone is not loadable."""
    import jevbench_run as JB
    jb = JB.jevbench(clone)
    pool = []
    for tier, t in JB.load_items(clone, jb):
        pool.append({"id": t.id, "tier": tier,
                     "state": JB.state_text(t.state),
                     "question": JB.to_question(t)})
    return pool


def labelled_cases(path: str, only_labelled: bool = True) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        cases = [json.loads(ln) for ln in f if ln.strip()]
    if only_labelled:
        import inject_labels
        lab = {k[0] for k in inject_labels.truth("rubric")}
        cases = [c for c in cases if c["case"] in lab]
    return cases


def build_plan(*, pool: list[dict] | None = None, cases: list[dict] | None
               = None, ks=KS, questions_fn=None, stage3_fn=None,
               limit: int | None = None, count=est_tokens) -> dict:
    """{calls: [call], skipped: {set: n}, sets: {set: {calls, k}}}: every
    call to make, in set order. `pool` None: no JevBench sets; `cases` None:
    no skill set. `limit`: the first N calls of each set."""
    calls, skipped = [], {}
    if pool:
        one = [{"set": "jevbench", "item": it["id"], "k": 1,
                "mode": "decide", "state": it["state"],
                "questions": [rename(it["question"], "q0")]} for it in pool]
        calls += one[:limit] if limit else one
        for k in ks:
            cs, sk = k_calls(pool, k, count)
            calls += cs[:limit] if limit else cs
            if sk:
                skipped[f"k{k}"] = sk
    if cases:
        cs = skill_calls(cases, questions_fn=questions_fn,
                         stage3_fn=stage3_fn)
        calls += cs[:limit] if limit else cs
    sets: dict = {}
    for c in calls:
        s = sets.setdefault(c["set"], {"calls": 0, "questions": 0})
        s["calls"] += 1
        s["questions"] += c["k"] + len(c.get("followup") or [])
    return {"calls": calls, "skipped": skipped, "sets": sets}


def request_counts(plan: dict) -> dict:
    """The exact number of model-server requests each arm makes, from the
    plan alone. Sequential: two reads a question (two orders), never fewer
    (a label outside the top K re-reads: not countable offline). Batched: one
    /decide-batch request a call -- a Turn call adds one more for the stage-3
    question and one release at the Turn's close -- plus the one-time
    derivation of each message structure (decider_batch: a few /apply-template
    and /tokenize requests, counted apart)."""
    out: dict = {}
    for c in plan["calls"]:
        n = c["k"] + len(c.get("followup") or [])
        o = out.setdefault(c["set"], {"calls": 0, "questions": 0,
                                      "sequential_reads": 0,
                                      "batch_requests": 0})
        o["calls"] += 1
        o["questions"] += n
        o["sequential_reads"] += 2 * n
        o["batch_requests"] += (1 + len(c.get("followup") or [])
                                + (1 if c["mode"] == "turn" else 0))
    return out


def estimate_minutes(plan: dict, count=est_tokens) -> dict:
    """The per-read arm's wall time by the brief's fit: ARITHMETIC. Per set:
    reads x 0.26 s + processed tokens x 2.3 ms, where a state is processed
    once per call and every read adds its question's tokens."""
    out: dict = {}
    for c in plan["calls"]:
        st = count(c["state"])
        qs = list(c["questions"]) + list(c.get("followup") or [])
        reads = 2 * len(qs)
        toks = st + 2 * sum(q_tokens(q, count) for q in qs)
        o = out.setdefault(c["set"], {"sequential_s": 0.0})
        o["sequential_s"] += reads * FIXED_S_PER_READ + toks * S_PER_TOKEN
    return {k: {"sequential_minutes": round(v["sequential_s"] / 60, 1),
                "basis": "the brief's fit (0.26 s/read + 2.3 ms/token), "
                         "arithmetic"} for k, v in out.items()}


# ------------------------------------------------------------ the compare ---
def probs_of(a: dict) -> dict:
    """A typed answer's distribution over its keys."""
    import decider_bonsai as D
    return {str(k): float(v) for k, v in D.answer_probs(a).items()}


def argmax_of(p: dict) -> str:
    return max(p, key=lambda k: (p[k], k))


def compare_answers(seq: dict, bat: dict) -> dict:
    """{argmax_differs, max_abs_p, tie_flip, same_keys}: one question's answer
    by the per-read path against the batched one."""
    ps, pb = probs_of(seq), probs_of(bat)
    keys = sorted(set(ps) | set(pb))
    ts = bool((seq.get("diagnostics") or {}).get("tie"))
    tb = bool((bat.get("diagnostics") or {}).get("tie"))
    return {"same_keys": set(ps) == set(pb),
            "argmax_differs": argmax_of(ps) != argmax_of(pb),
            "max_abs_p": max((abs(ps.get(k, 0.0) - pb.get(k, 0.0))
                              for k in keys), default=0.0),
            "tie_flip": ts != tb}


def compare_call(seq_answers: list[dict], bat_answers: list[dict]) -> dict:
    cs = [compare_answers(a, b) for a, b in zip(seq_answers, bat_answers)]
    return {"questions": len(cs),
            "argmax_differs": sum(c["argmax_differs"] for c in cs),
            "max_abs_p": max((c["max_abs_p"] for c in cs), default=0.0),
            "tie_flips": sum(c["tie_flip"] for c in cs),
            "keys_differ": sum(not c["same_keys"] for c in cs)}


def _pct(xs: list, q: float):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    k = (len(xs) - 1) * q
    f, c = math.floor(k), math.ceil(k)
    return xs[f] if f == c else xs[f] * (c - k) + xs[c] * (k - f)


def summarize(rows: list[dict]) -> dict:
    """The table: per set, calls run, wall seconds per call and per question
    for each arm (median, p90), the speedup (sum of sequential wall over sum
    of batched wall, and the median of the per-call ratios), the identity
    counts and the fallbacks of the batched arm."""
    import decider_bonsai as D
    out: dict = {"tie_band": D.TIE_BAND, "sets": {}}
    for s in sorted({r["set"] for r in rows}):
        rs = [r for r in rows if r["set"] == s and r.get("ok")]
        if not rs:
            continue
        seq = [r["seq"]["wall_s"] for r in rs]
        bat = [r["batch"]["wall_s"] for r in rs]
        qn = [r["questions"] for r in rs]
        ratios = [a / b for a, b in zip(seq, bat) if b > 0]
        cmpr = [r["compare"] for r in rs]
        maxp = max(c["max_abs_p"] for c in cmpr)
        out["sets"][s] = {
            "calls": len(rs), "k": sorted({r["k"] for r in rs}),
            "questions": sum(qn),
            "seq_s_per_call": {"median": statistics.median(seq),
                               "p90": _pct(seq, 0.9)},
            "batch_s_per_call": {"median": statistics.median(bat),
                                 "p90": _pct(bat, 0.9)},
            "seq_s_per_question": statistics.median(
                [a / n for a, n in zip(seq, qn)]),
            "batch_s_per_question": statistics.median(
                [a / n for a, n in zip(bat, qn)]),
            "speedup_total": (sum(seq) / sum(bat)) if sum(bat) else None,
            "speedup_median": statistics.median(ratios) if ratios else None,
            "argmax_differs": sum(c["argmax_differs"] for c in cmpr),
            "max_abs_p": maxp,
            "max_abs_p_over_tie_band": maxp / D.TIE_BAND,
            "tie_flips": sum(c["tie_flips"] for c in cmpr),
            "batch_fallbacks": sum(1 for r in rs
                                   if r["batch"].get("read_paths") != ["batch"]),
            "failed_calls": sum(1 for r in rows
                                if r["set"] == s and not r.get("ok"))}
    return out


def table(summary: dict) -> str:
    lines = [f"{'set':<10}{'calls':>6}{'K':>9}{'seq s/call':>12}"
             f"{'batch s/call':>14}{'seq s/q':>9}{'batch s/q':>11}"
             f"{'speedup':>9}{'argmax!=':>10}{'max|dp|':>10}{'ties':>6}"
             f"{'fallbk':>8}"]
    for s, r in summary["sets"].items():
        lines.append(
            f"{s:<10}{r['calls']:>6}{','.join(map(str, r['k'])):>9}"
            f"{r['seq_s_per_call']['median']:>12.2f}"
            f"{r['batch_s_per_call']['median']:>14.2f}"
            f"{r['seq_s_per_question']:>9.2f}{r['batch_s_per_question']:>11.2f}"
            f"{(r['speedup_total'] or 0):>9.2f}{r['argmax_differs']:>10}"
            f"{r['max_abs_p']:>10.4f}{r['tie_flips']:>6}"
            f"{r['batch_fallbacks']:>8}")
    lines.append(f"(max|dp| against the batch-noise floor TIE_BAND "
                 f"{summary['tie_band']})")
    return "\n".join(lines)


# --------------------------------------------------------------- the run ----
def _set_batch(on: bool) -> None:
    if on:
        os.environ["YAMADORI_DECIDER_BATCH"] = "1"
    else:
        os.environ.pop("YAMADORI_DECIDER_BATCH", None)


def _read_path_of(answers: list[dict]) -> list[str]:
    return sorted({(a.get("diagnostics") or {}).get("read_path")
                   or "sequential" for a in answers})


def _arm_decide(call: dict, on: bool, timeout: float = 600) -> dict:
    """One arm of a `decide` call: D.decide over the call's state."""
    import decider_bonsai as D
    _set_batch(on)
    t0 = time.perf_counter()
    out = D.decide(call["state"], call["questions"], timeout=timeout)
    wall = time.perf_counter() - t0
    names = [q["name"] for q in call["questions"]]
    answers = [out["answers"][n] for n in names]
    return _record(answers, wall, out.get("usage"))


def _arm_turn(call: dict, on: bool) -> dict:
    """One arm of a `turn` call: a Turn, decide(stage 2), decide(stage 3)."""
    import decide_turn as T
    _set_batch(on)
    t0 = time.perf_counter()
    with T.Turn([], state=call["state"], state_info=call["state_info"],
                on=True, key=f"batch-ab:{call['item']}",
                request=call["item"]) as t:
        answers = t.decide(call["questions"])
        answers += t.decide(call["followup"])
    wall = time.perf_counter() - t0
    return _record(answers, wall, None)


def _record(answers: list[dict], wall: float, usage) -> dict:
    d = [a.get("diagnostics") or {} for a in answers]
    b = next((x.get("batch") for x in d if x.get("batch")), None)
    return {"wall_s": round(wall, 4), "read_paths": _read_path_of(answers),
            "reads": sum(int(x.get("reads") or 0) for x in d),
            "processed_tokens": sum(int(x.get("processed_tokens") or 0)
                                    for x in d),
            "prompt_tokens": sum(int(x.get("prompt_tokens") or 0)
                                 for x in d),
            "usage": usage, "engine": (b or {}).get("timings"),
            "_answers": answers}


def run_call(call: dict, index: int) -> dict:
    """Both arms of one call, the order alternating by `index`, and their
    comparison. Never raises: a failed call is recorded."""
    arm = _arm_turn if call["mode"] == "turn" else _arm_decide
    order = ("seq", "batch") if index % 2 == 0 else ("batch", "seq")
    res: dict = {}
    row = {"v": RUN_VERSION, "set": call["set"], "item": call["item"],
           "k": call["k"], "questions": call["k"]
           + len(call.get("followup") or []), "order": list(order)}
    try:
        for name in order:
            res[name] = arm(call, name == "batch")
    except Exception as e:                                       # noqa: BLE001
        row.update(ok=False, error=f"{type(e).__name__}: {e}"[:300])
        return row
    cmp = compare_call(res["seq"].pop("_answers"),
                       res["batch"].pop("_answers"))
    row.update(ok=True, seq=res["seq"], batch=res["batch"], compare=cmp)
    return row


def run(plan: dict, run_dir: str, *, preflight=None, log=print) -> list[dict]:
    os.makedirs(run_dir, exist_ok=True)
    rows = []
    path = os.path.join(run_dir, "items.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for i, call in enumerate(plan["calls"]):
            if preflight and i % CHECK_EVERY == 0:
                preflight()
            row = run_call(call, i)
            rows.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            if (i + 1) % 10 == 0:
                log(f"  {i + 1}/{len(plan['calls'])}")
    return rows


def probe(log=print) -> dict:
    """One tiny batched call: the engine must answer it as a batch, else
    there is nothing to compare (exit 3)."""
    import decider_batch as B
    import decider_bonsai as D
    B.reset()
    _set_batch(True)
    a = B.read_many("Probe: the sky is blue.", [D.q_noul("probe", "Is it?")])
    paths = _read_path_of(a)
    return {"ok": paths == ["batch"], "read_paths": paths, "stats": B.stats()}


def plan_text(plan: dict, state: dict) -> str:
    cnt = request_counts(plan)
    est = estimate_minutes(plan)
    lines = ["THE PLAN (nothing is sent without --run --gpu-go):"]
    for s, o in cnt.items():
        lines.append(
            f"  {s:<9} {o['calls']:>4} calls, {o['questions']:>5} questions: "
            f"{o['sequential_reads']:>6} per-read requests (at least) vs "
            f"{o['batch_requests']:>5} /decide-batch requests; per-read arm "
            f"~{est[s]['sequential_minutes']} min by the brief's fit")
    if plan["skipped"]:
        lines.append(f"  left out (over the 64k-token request limit, by "
                     f"chars/3): {plan['skipped']}")
    for k, v in state.items():
        lines.append(f"  {k}: {v}")
    lines += [
        f"  reader: {DEFAULT_MODEL} (layout v3's jjava reader), decider_target",
        "  each item runs BOTH arms, the first alternating by item; compared:"
        " argmax, max |p|, tie flips",
        "  writes: bench/decider/results/batch_ab/<run>/{items.jsonl, "
        "summary.json}",
        "  command (the operator says GPU go):",
        "    C:\\Users\\jwals\\textgen\\installer_files\\env\\python.exe "
        "bench/decider/batch_ab.py --run --gpu-go"]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", action="store_true",
                    help="run it (GPU); refused without --gpu-go")
    ap.add_argument("--gpu-go", action="store_true",
                    help="the operator's 'GPU go' for this run")
    ap.add_argument("--plan-json", action="store_true")
    ap.add_argument("--jevbench", help="a JevBench clone (default: "
                    "jevbench_run's)")
    ap.add_argument("--cases", default=DEFAULT_CASES,
                    help="the skill window's cases.jsonl")
    ap.add_argument("--all-cases", action="store_true",
                    help="every case, not only the rubric-labelled ones")
    ap.add_argument("--sets", default="jevbench,k,skill",
                    help="comma list of jevbench, k, skill")
    ap.add_argument("--ks", default=",".join(map(str, KS)))
    ap.add_argument("--limit", type=int, help="the first N calls of each set")
    import decider_target as DT
    DT.add_args(ap)
    a = ap.parse_args(argv)
    if a.run and not a.gpu_go:
        print("REFUSED: --run needs --gpu-go (the operator says 'GPU go'); "
              "nothing was sent")
        return 2
    sets = {s.strip() for s in a.sets.split(",") if s.strip()}
    ks = tuple(int(x) for x in a.ks.split(",") if x.strip())
    state: dict = {}
    pool = cases = None
    if sets & {"jevbench", "k"}:
        import jevbench_run as JB
        clone = os.path.abspath(a.jevbench or JB.DEFAULT_CLONE)
        try:
            pool = jevbench_pool(clone)
            state["jevbench"] = f"{len(pool)} public items from {clone}"
        except Exception as e:                                   # noqa: BLE001
            state["jevbench"] = (f"NOT LOADABLE from {clone}: "
                                 f"{type(e).__name__}: {e}"[:200])
    if "skill" in sets:
        try:
            cases = labelled_cases(a.cases, not a.all_cases)
            state["skill cases"] = f"{len(cases)} from {a.cases}"
        except Exception as e:                                   # noqa: BLE001
            state["skill cases"] = (f"NOT LOADABLE from {a.cases}: "
                                    f"{type(e).__name__}: {e}"[:200])
    plan = build_plan(pool=pool if "jevbench" in sets or "k" in sets else None,
                      cases=cases, ks=ks if "k" in sets else (),
                      limit=a.limit)
    if "jevbench" not in sets and pool:
        plan["calls"] = [c for c in plan["calls"] if c["set"] != "jevbench"]
    if a.plan_json:
        print(json.dumps({"counts": request_counts(plan),
                          "estimate": estimate_minutes(plan),
                          "skipped": plan["skipped"], "state": state},
                         indent=1))
    else:
        print(plan_text(plan, state))
    if not a.run:
        return 0
    if not plan["calls"]:
        print("NOT RUN: the plan is empty")
        return 3
    from bonsai_decider import Busy
    a.model = a.model or DEFAULT_MODEL
    try:
        target = DT.from_args(a)
    except (DT.NotRun, ValueError) as e:
        print(f"NOT RUN: {e}")
        return 3
    DT.install(target, decisions_dir=os.path.join(ROOT, "logs",
                                                  "decider_measure"))
    run_dir = os.path.join(OUT_DIR, f"{target.model}-"
                           + time.strftime("%Y%m%d-%H%M%S"))
    t0 = time.time()
    try:
        DT.preflight(target)
        pr = probe()
        if not pr["ok"]:
            print("NOT RUN: the capability probe was not answered as a batch "
                  f"({json.dumps(pr)[:400]}): the engine needs the "
                  "decide-batch build (--decide-seqs N, --kv-unified)")
            return 3
        rows = run(plan, run_dir, preflight=lambda: DT.preflight(target))
    except Busy as e:
        print(f"STOPPED (not run): {e}")
        return 2
    finally:
        _set_batch(False)
    s = summarize(rows)
    s.update(version=RUN_VERSION, target=target.describe(),
             seconds=round(time.time() - t0, 1), calls=len(rows),
             failed=sum(1 for r in rows if not r.get("ok")))
    with open(os.path.join(run_dir, "summary.json"), "w",
              encoding="utf-8") as f:
        json.dump(s, f, indent=1)
    print(table(s))
    print("->", run_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
