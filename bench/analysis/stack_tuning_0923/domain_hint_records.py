"""What the domain suite actually recorded about hints, next to the offline
prediction from hint_trace.json. No service calls."""
import json, glob, os, collections
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
tr = json.load(open(os.path.join(HERE, "data", "hint_trace.json"), encoding="utf-8"))
pred = {p["id"]: p for p in tr["prompts"]}
qs = {q["question_id"][:8]: q for q in json.load(open(os.path.join(HERE, "data", "lb_questions_slim.json")))}
for p in tr["prompts"]:
    if p["set"] == "lb" and p["prod"]["by_floor"]["0.40"]["injected"] > 0:
        print("LB@0.40", p["id"], p["group"], qs[p["id"]].get("question_title"), p["prod"]["top5"][0]["score"], "|", p["prod"]["top5"][0]["recipe"][:100])
rows = []
for f in glob.glob(os.path.join(ROOT, "bench", "domain", "results", "*", "rows.jsonl")):
    for l in open(f, encoding="utf-8"):
        if l.strip():
            r = json.loads(l); r["_dir"] = os.path.basename(os.path.dirname(f)); rows.append(r)
print("domain rows", len(rows))
c = collections.Counter()
for r in rows:
    x = r.get("x_yamadori") or {}
    sel = x.get("selection") or {}
    h = x.get("hints") or []
    c[(r.get("arm"), bool(sel.get("hints")), len(h))] += 1
    if sel.get("hints"):
        p = pred.get(r.get("task"))
        print(f"{r['_dir']:28} {r.get('arm'):4} {r.get('task'):26} prod-injected {len(h)} {[(y.get('score'), y.get('source')) for y in h]} | offline@0.55 {p and p['prod']['by_floor']['0.55']['injected']} maxcos {p and p['prod']['max_cos']} | {r.get('outcome')}")
for k, v in sorted(c.items(), key=str):
    print(k, v)
