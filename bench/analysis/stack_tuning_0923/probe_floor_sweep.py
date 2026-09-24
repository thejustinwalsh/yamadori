"""Floor sweep on the 89 labelled hint probes (bench/hint_probes.jsonl), the
production path (served cache, task_domains, bucket collapse, k=3). For each
floor: probes where the correct bucket member is shown, where a wrong sibling
is shown, where nothing of the bucket is shown, and mean hints shown. One
batched embeddings call per 16 probes; no chat model."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "mcp"))
import numpy as np
import code_search as cs
import hints
probes = [json.loads(x) for x in open(os.path.join(ROOT, "bench", "hint_probes.jsonl"), encoding="utf-8") if x.strip()]
buckets = {}
for x in open(hints.BUCKETS, encoding="utf-8"):
    if x.strip():
        b = json.loads(x); buckets[b["bucket_id"]] = {m["member_id"]: m["recipe"].strip() for m in b["members"]}
rows, mat = hints.load()
qv = []
texts = [cs.QUERY_INSTRUCT + p["problem"] for p in probes]
for i in range(0, len(texts), 16):
    d = cs._post("/v1/embeddings", {"model": cs.EMBED_MODEL, "input": texts[i:i + 16]})
    v = np.array([r["embedding"] for r in d["data"]], dtype=np.float32)
    qv.append(v / np.linalg.norm(v, axis=1, keepdims=True))
qv = np.vstack(qv)
out = {}
for f in (0.40, 0.45, 0.50, 0.55, 0.60):
    c = {"n": len(probes), "any_hint": 0, "correct": 0, "wrong_sibling": 0, "bucket_silent": 0, "hints": 0,
         "correct_only": 0}
    for p, q in zip(probes, qv):
        members = buckets[p["bucket_id"]]; right = members[p["correct"]]
        task = hints.task_domains([{"role": "user", "content": p["problem"]}])
        got = hints._select_vector(rows, mat, q, hints.TOP_K, f, True, task)
        t = [h["recipe"].strip() for h in got]
        mine = [x for x in t if x in members.values()]
        c["any_hint"] += bool(got); c["hints"] += len(got)
        c["correct"] += right in t; c["wrong_sibling"] += any(x != right for x in mine)
        c["bucket_silent"] += not mine; c["correct_only"] += mine == [right]
    c["precision_when_bucket_shown"] = round(c["correct"] / max(1, c["n"] - c["bucket_silent"]), 3)
    out[f"{f:.2f}"] = c
    print(f, c)
json.dump(out, open(os.path.join(HERE, "data", "probe_floor_sweep.json"), "w"), indent=1)
