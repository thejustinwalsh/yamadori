"""Summarise data/hint_trace.json (no service calls)."""
import json, os, collections, statistics as st
HERE = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(HERE, "data", "hint_trace.json"), encoding="utf-8"))
P = d["prompts"]
def sets():
    yield "lb (21)", [p for p in P if p["set"] == "lb"]
    for g in ("LCB_generation", "coding_completion"):
        yield f"  lb {g}", [p for p in P if p["set"] == "lb" and p["group"] == g]
    yield "domain (120)", [p for p in P if p["set"] == "domain"]
    for g in ("typescript", "typegpu", "rust_wasm", "react"):
        yield f"  domain {g}", [p for p in P if p["set"] == "domain" and p["group"] == g]
for var in ("prod", "recipe_q", "lb_core"):
    print(f"\n=== variant {var}")
    print(f"{'set':26}{'n':>4}{'filt':>6}{'maxcos med':>11}{'[min,max]':>16}   injected>=1 at floor " + " ".join(d_ for d_ in ("0.40","0.45","0.50","0.55","0.60")) + "   stage@0.55 (floor/domain/inj)   mean hints@0.55  supp@0.55")
    for name, ps in sets():
        ps = [p for p in ps if var in p]
        if not ps: continue
        mc = [p[var]["max_cos"] for p in ps]
        filt = sum(1 for p in ps if p["task_domains"] is not None)
        inj = [sum(1 for p in ps if p[var]["by_floor"][f]["injected"] > 0) for f in ("0.40","0.45","0.50","0.55","0.60")]
        stc = collections.Counter(p[var]["by_floor"]["0.55"]["stage"].split(":")[0] for p in ps)
        mh = st.mean(p[var]["by_floor"]["0.55"]["injected"] for p in ps)
        sup = sum(p[var]["by_floor"]["0.55"]["suppressed_by_bucket"] for p in ps)
        print(f"{name:26}{len(ps):>4}{filt:>6}{st.median(mc):>11.3f}{'[%.3f,%.3f]'%(min(mc),max(mc)):>16}   " + "  ".join(f"{x:>3}" for x in inj) + f"      {stc.get('floor',0)}/{stc.get('domain gate',0)}/{stc.get('injected',0)}      {mh:.2f}     {sup}")
# per-floor full stage counts, prod
print("\n=== prod: stage per floor, all 141")
for f in ("0.40","0.45","0.50","0.55","0.60"):
    for s in ("lb","domain"):
        ps=[p for p in P if p["set"]==s]
        c=collections.Counter(p["prod"]["by_floor"][f]["stage"].split(":")[0] for p in ps)
        mh=st.mean(p["prod"]["by_floor"][f]["injected"] for p in ps)
        sup=sum(p["prod"]["by_floor"][f]["suppressed_by_bucket"] for p in ps)
        nc=sum(1 for p in ps if p["prod"]["by_floor"][f]["injected"]!=p["prod"]["by_floor"][f]["injected_no_collapse"])
        print(f"  floor {f} {s:7} n={len(ps):3}  nothing>=floor {c.get('floor',0):3}  domain-gate {c.get('domain gate',0):3}  injected {c.get('injected',0):3}  mean hints {mh:.2f}  suppressed rows {sup}  prompts where collapse changed count {nc}")
print("\n=== task domains (strong evidence) ")
for s in ("lb","domain"):
    ps=[p for p in P if p["set"]==s]
    print(s, collections.Counter(str(p["task_domains"]) for p in ps).most_common(12))
print("\n=== files of top-1 row (prod) per set")
for s in ("lb","domain"):
    ps=[p for p in P if p["set"]==s]
    print(s, collections.Counter(p["prod"]["top5"][0]["file"] for p in ps).most_common(10))
print("\n=== files injected at 0.45 / 0.50 (prod)")
for f in ("0.45","0.50","0.55"):
  for s in ("lb","domain"):
    ps=[p for p in P if p["set"]==s]
    print(f, s, collections.Counter(x for p in ps for x in p["prod"]["by_floor"][f]["files"]).most_common(10))
