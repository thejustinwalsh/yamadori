"""Fact-check probe (docs/LAYA-FACTCHECK.md): the 120 held-out rows through the
LIVE Laya service (/route, engine=trained), then selection.decide with that
signal handed in -- the combination production actually runs.
heldout_rows.jsonl is the per-row output of bench/eval_route_heldout.py from
the same fact-check; live_route_check.txt is the recorded result.
Prints aggregates only, never question text."""
import json, os, sys, tempfile, time, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "mcp"))
tmp = tempfile.mkdtemp(prefix="factcheck_")
os.environ.setdefault("YAMADORI_CORPUS_DB", os.path.join(tmp, "corpus.sqlite3"))
os.environ.setdefault("LLAMA_STACK_URL", "http://127.0.0.1:1")
import selection, deps, domains, tiers  # noqa: E402,F401

HERE = os.path.dirname(os.path.abspath(__file__))
rows = [json.loads(x) for x in open(os.path.join(ROOT, "bench", "laya_routing_heldout_packages.jsonl"), encoding="utf-8") if x.strip()]
offline = [json.loads(x) for x in open(os.path.join(HERE, "heldout_rows.jsonl"), encoding="utf-8")]
y = [r["label"] for r in rows]
yb = [l == "investigate" for l in y]
hard = [i for i, r in enumerate(rows) if r.get("hard")]

live, ms = [], []
for r in rows:
    body = json.dumps({"task": "route_in", "engine": "trained", "question": r["question"],
                       "context": r.get("context", "")}).encode()
    req = urllib.request.Request("http://127.0.0.1:1237/route", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=30) as fh:
        d = json.loads(fh.read().decode("utf-8"))
    ms.append((time.time() - t0) * 1000)
    live.append(d)

agree = sum(1 for d, o in zip(live, offline) if d.get("choice") == o["new"])
print(f"live /route (trained) vs offline new-head argmax: {agree}/120 identical choices")
print("live engines:", {e: sum(1 for d in live if d.get("engine") == e) for e in {d.get('engine') for d in live}})
print(f"live abstain: {sum(1 for d in live if d.get('abstain'))}/120  (gate {live[0].get('gate')})")
lb = [d.get("choice") == "investigate" for d in live]
print(f"live head investigate-vs-not (argmax): {sum(p == t for p, t in zip(lb, yb))}/120")
srt = sorted(ms)
print(f"round-trip ms: median {srt[60]:.0f}, p90 {srt[108]:.0f}")

t = tiers.resolve({"reasoning_effort": "max"}, tiers.from_header(None))
dbs = selection.symbol_dbs()
sel_alone, sel_with = [], []
esc = 0
for r, d in zip(rows, live):
    msgs = ([{"role": "user", "content": r["context"]}] if r.get("context") else []) \
        + [{"role": "user", "content": r["question"]}]
    gate = domains.tool_admission(msgs, None)
    a = selection.decide(msgs, t, gate, dbs=dbs)
    keep = {k: d.get(k) for k in ("choice", "probabilities", "margin", "abstain", "gate", "engine")}
    b = selection.decide(msgs, t, gate, laya=keep, laya_status="answered", dbs=dbs)
    sel_alone.append(bool(a["investigate"]))
    sel_with.append(bool(b["investigate"]))
    if b["investigate"] and not a["investigate"]:
        esc += 1
def acc(p, idx=None):
    idx = range(120) if idx is None else idx
    return sum(p[i] == yb[i] for i in idx), len(list(idx))
print("selection.decide, Laya absent: %d/%d" % acc(sel_alone))
print("selection.decide, live Laya signal (disagreement escalates): %d/%d" % acc(sel_with))
print("  hard slice alone %d/%d, with Laya %d/%d" % (acc(sel_alone, hard) + acc(sel_with, hard)))
print(f"  rows where Laya flipped the decision to investigate: {esc}")
fp_a = sum(1 for p, t in zip(sel_alone, yb) if p and not t); fn_a = sum(1 for p, t in zip(sel_alone, yb) if t and not p)
fp_w = sum(1 for p, t in zip(sel_with, yb) if p and not t); fn_w = sum(1 for p, t in zip(sel_with, yb) if t and not p)
print(f"  missed investigations (should investigate, did not): alone {fn_a}, with Laya {fn_w}")
print(f"  unneeded investigations: alone {fp_a}, with Laya {fp_w}")
