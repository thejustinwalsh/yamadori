"""Q1: where does each real prompt drop out of the hint pipeline?

Production path, offline: the served cache index/hints.npz (signature checked,
never rebuilt here), hints.task_domains, domains.eligible, the bucket collapse
in hints._select_vector. The ONLY service call is the embeddings model on
llama-swap :11434, through code_search.embed -- one batched call per prompt set
per variant. No chat model, nothing to :1234.

Prompts:
  lb      the 21 LiveBench coding questions actually run (order_coding.json of
          lb-20260923-minp0), user turn = question turns[0], as LiveBench sends it
  domain  the 120 uncontaminated domain-suite prompts (tasks.jsonl minus
          three_tsl, plus tasks_react.jsonl), user turn = task["prompt"], as
          bench/domain/run.py sends it

Variants (each one embed call per prompt):
  prod      QUERY_INSTRUCT (production) + the whole user turn    <- what serves
  recipe_q  a recipe-shaped instruction + the whole user turn     (diagnostic)
  lb_core   production instruction + the LiveBench problem text only, the
            "### Instructions" / "### Format" / "### Answer" boilerplate cut
            (diagnostic, lb only)

Writes data/hint_trace.json. Run with the stack interpreter.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "mcp"))

import numpy as np  # noqa: E402

import code_search as cs  # noqa: E402
import domains  # noqa: E402
import hints  # noqa: E402

FLOORS = (0.40, 0.45, 0.50, 0.55, 0.60)
RECIPE_INSTRUCT = ("Instruct: Given a programming task, retrieve short "
                   "engineering notes and recipes that apply to solving it\n"
                   "Query: ")


def lb_prompts():
    qs = json.load(open(os.path.join(HERE, "data", "lb_questions_slim.json"),
                        encoding="utf-8"))
    by = {q["question_id"]: q for q in qs}
    order = json.load(open(os.path.join(
        ROOT, "bench", "livebench", "results", "lb-20260923-minp0",
        "order_coding.json"), encoding="utf-8"))
    ids = [r["question_id"] for r in (order if isinstance(order, list)
                                      else order.get("order", order.get("rows")))]
    ran = {json.loads(line)["id"] for line in open(os.path.join(
        ROOT, "bench", "livebench", "results", "lb-20260923-minp0",
        "rows_yamadori-xhigh_coding.jsonl"), encoding="utf-8") if line.strip()}
    out = []
    for qid in ids:
        if qid in ran:
            q = by[qid]
            out.append({"id": qid[:8], "set": "lb", "group": q["task"],
                        "text": q["turns"][0]})
    return out


def domain_prompts():
    out = []
    for fn in ("tasks.jsonl", "tasks_react.jsonl"):
        for line in open(os.path.join(ROOT, "bench", "domain", fn),
                         encoding="utf-8"):
            if not line.strip():
                continue
            t = json.loads(line)
            if t["contaminated"]:
                continue
            out.append({"id": t["id"], "set": "domain", "group": t["domain"],
                        "text": t["prompt"]})
    return out


def lb_core(text: str) -> str:
    """The problem statement without LiveBench's wrapper sections."""
    m = re.search(r"### Question:\s*(.*?)(?:\n### Format:|\n### Answer:|$)",
                  text, re.S)
    return (m.group(1) if m else text).strip()


def trace_one(rows, mat, q, task, floors=FLOORS):
    sims = mat @ q
    elig = np.array([hints._eligible(r, task) for r in rows])
    order = np.argsort(-sims, kind="stable")
    top = [{"score": round(float(sims[i]), 4),
            "file": rows[i]["_file"],
            "eligible": bool(elig[i]),
            "recipe": rows[i]["recipe"][:110]} for i in order[:5]]
    out = {"max_cos": round(float(sims.max()), 4),
           "max_cos_eligible": round(float(sims[elig].max()), 4) if elig.any() else None,
           "n_eligible_rows": int(elig.sum()),
           "top5": top, "by_floor": {}}
    for f in floors:
        above = int((sims >= f).sum())
        above_el = int(((sims >= f) & elig).sum())
        got = hints._select_vector(rows, mat, q, hints.TOP_K, f, True, task)
        nocol = hints._select_vector(rows, mat, q, hints.TOP_K, f, False, task)
        sup = sum(len(h.get("_suppressed") or []) for h in got)
        if above == 0:
            stage = "floor: nothing in the corpus clears it"
        elif above_el == 0:
            stage = "domain gate: rows clear the floor, none eligible"
        else:
            stage = "injected"
        out["by_floor"][f"{f:.2f}"] = {
            "above_floor": above, "above_floor_eligible": above_el,
            "injected": len(got), "injected_no_collapse": len(nocol),
            "suppressed_by_bucket": sup, "stage": stage,
            "files": [h["_file"] for h in got]}
    return out


def main():
    rows, mat = hints.load()
    texts = [hints._hint_text(r) for r in rows]
    z = np.load(hints.CACHE)
    assert str(z["sig"]) == hints._signature(texts), "hint cache is stale"
    assert int((np.abs(mat).sum(1) == 0).sum()) == 0, "zero vectors in cache"

    prompts = lb_prompts() + domain_prompts()
    print(f"{len(prompts)} prompts; corpus {len(rows)} rows")
    calls = 0
    for p in prompts:
        msgs = [{"role": "user", "content": p["text"]}]
        p["task_domains"] = (sorted(hints.task_domains(msgs))
                             if hints.task_domains(msgs) is not None else None)
        p["strong"] = sorted(domains.strong_evidence(msgs))
        p["words"] = sorted(domains.detect(msgs))
        p["chars"] = len(p["text"])

    variants = {"prod": (cs.QUERY_INSTRUCT, lambda p: p["text"], None),
                "recipe_q": (RECIPE_INSTRUCT, lambda p: p["text"], None),
                "lb_core": (cs.QUERY_INSTRUCT, lambda p: lb_core(p["text"]), "lb")}
    t0 = time.time()
    for name, (inst, fn, only) in variants.items():
        sel = [p for p in prompts if only is None or p["set"] == only]
        qtexts = [inst + fn(p) for p in sel]
        vecs = []
        for i in range(0, len(qtexts), 16):
            d = cs._post("/v1/embeddings", {"model": cs.EMBED_MODEL,
                                            "input": qtexts[i:i + 16]})
            v = np.array([r["embedding"] for r in d["data"]], dtype=np.float32)
            vecs.append(v / np.clip(np.linalg.norm(v, axis=1, keepdims=True),
                                    1e-9, None))
            calls += 1
        V = np.vstack(vecs)
        assert V.shape[0] == len(sel) and np.all(np.abs(np.linalg.norm(V, axis=1) - 1) < 1e-3)
        for p, q in zip(sel, V):
            task = (set(p["task_domains"]) if p["task_domains"] is not None
                    else None)
            p[name] = trace_one(rows, mat, q, task)
    # parity check: the production variant must equal hints.select() itself
    # for a few prompts (one embed each, through the real select()).
    parity = []
    for p in prompts[:3] + prompts[-3:]:
        got = hints.select(p["text"])
        calls += 1
        parity.append(len(got) == p["prod"]["by_floor"]["0.55"]["injected"])
    print(f"embedding requests: {calls} (batched), prompts embedded: "
          f"{sum(1 for p in prompts for v in variants if v in p) + 6}, "
          f"{time.time() - t0:.1f}s; parity with hints.select: {parity}")
    json.dump({"floors": FLOORS, "k": hints.TOP_K, "corpus_rows": len(rows),
               "prompts": prompts, "parity": parity},
              open(os.path.join(HERE, "data", "hint_trace.json"), "w",
                   encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
