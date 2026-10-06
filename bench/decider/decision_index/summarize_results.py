"""Per-benchmark status / time table for a Decision Index results.jsonl produced through jjava_engine.

  python summarize_results.py RUN_DIR/results.jsonl [--md]
Columns: n, ok / unsupported / error, request ms (median, max), ms per field, seconds waited on 429/529 (adapter),
and which model answered (response.model; x_yamadori.jjava.model). Wall time per row includes waited seconds."""
import collections, json, statistics, sys

rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8") if l.strip()]
by = collections.defaultdict(list)
for r in rows:
    by[(r["catalog_id"], r["dataset"])].append(r)


def fields(r):
    p = r.get("payload")
    if p:
        return len(p["questions"])
    return len((r.get("response") or {}).get("answers") or {}) or None


print("cat dataset n ok unsup err ms_med ms_max ms_per_field waited_s answered_by")
tot = collections.Counter()
for (cid, name), rs in sorted(by.items()):
    st = collections.Counter(r["status"] for r in rs)
    ms = [r["total_wall_ms"] for r in rs]
    pf = [r["total_wall_ms"] / f for r in rs if (f := fields(r))]
    waited = sum(((r.get("response") or {}).get("x_adapter") or {}).get("waited_s", 0) for r in rs)
    models = collections.Counter((r.get("response") or {}).get("model") for r in rs if r["status"] == "ok")
    print(cid, name, len(rs), st["ok"], st["unsupported"], st["error"], round(statistics.median(ms)), round(max(ms)), round(statistics.median(pf)) if pf else "-", waited, dict(models))
    tot.update(n=len(rs), ok=st["ok"], unsupported=st["unsupported"], error=st["error"])
allms = [r["total_wall_ms"] for r in rows]
nf = sum(f for r in rows if (f := fields(r)))
print("total", dict(tot), "median ms/request", round(statistics.median(allms)), "sum s", round(sum(allms) / 1000, 1), "fields", nf, "mean ms/field", round(sum(allms) / max(nf, 1)))
for r in rows:
    if r["status"] != "ok":
        print(r["status"], r["run_id"][:90], (r.get("error") or "")[:200].replace("\n", " "))
