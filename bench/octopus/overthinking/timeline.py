import json,sys,statistics
from common import OUT
sys.stdout.reconfigure(encoding='utf-8')
R=json.load(open(OUT('requests.json')))
CPT=3.88  # reasoning chars per token, calibrated on v0e1 (601,362 chars / 154,970 tok)
def rtok(r):
    if r['deep_fire'] and r['inv_s']: return None  # second brain's trace streamed as reasoning
    if r['reasoning_tok']: return r['reasoning_tok']
    return round((r['reasoning_chars'] or 0)/CPT)
for key in ('pilot','v0b','v0e1','v0e2'):
    rows=R[key]; t0=rows[0]['t0']; t1=max(r['t_end'] for r in rows)
    wall=t1-t0
    req=sum(r['dur'] for r in rows)
    deep=sum(r['inv_s'] or 0 for r in rows if r['deep_fire'])
    comp_t=sum(r['dur'] for r in rows if r['compaction'])
    util_t=sum(r['dur'] for r in rows if r['utility'] and not r['compaction'])
    model=sum((r['model_ms'] or 0)/1000 for r in rows)
    fix_t=sum(max(0,r['dur']-(r['inv_s'] or 0)-(r['model_ms'] or 0)/1000) for r in rows if r['tc_fixup'])
    fail_t=sum(r['dur'] for r in rows if r['status']!=200)
    gaps=[]
    srt=sorted(rows,key=lambda r:r['t0'])
    for a,b in zip(srt,srt[1:]):
        gaps.append(b['t0']-a['t_end'])
    print(f"== {key}: wall {wall/3600:.2f} h ({wall:.0f}s). in requests {req:.0f}s; deep {deep:.0f}s; compaction reqs {comp_t:.0f}s; utility {util_t:.0f}s; main model_ms {model:.0f}s; fixup-ish {fix_t:.0f}s; non-200 {fail_t:.0f}s; gaps(client/tools) {sum(g for g in gaps if g>0):.0f}s (max gap {max(gaps):.0f}s, overlapping {sum(g for g in gaps if g<0):.0f})")
    main=[r for r in rows if r['route']=='agent_step' and not r['deep_fire'] and r['status']==200]
    rts=[rtok(r) for r in main if rtok(r) is not None]
    comps=[r['completion'] or 0 for r in main]
    if rts:
        q=statistics.quantiles(rts,n=10)
        print(f"   agent_step main n={len(rts)} reasoning tok: median {statistics.median(rts):.0f} p90 {q[8]:.0f} max {max(rts)} total {sum(rts)}; completion total {sum(comps)} ; reasoning share of completion {sum(rts)/max(1,sum(comps)):.2f}")
        for cap in (2048,3072,4096,6144):
            over=[x for x in rts if x>cap]
            print(f"     > {cap}: {len(over)} steps, {sum(x-cap for x in over)} tokens above cap")
        if key.startswith('v0e'):
            nud=[x for x in rts if x>=0.6*6144]; hard=[x for x in rts if x>=6144-20]
            print(f"     nudge reached (>=3686): {len(nud)}; at hard cap (>=6124): {len(hard)}")
    # rate
    dec=[(r['completion'] or 0)/((r['model_ms'] or 1)/1000) for r in main if r['model_ms'] and r['completion'] and r['completion']>300]
    if dec: print(f"   tok/s (completion/model_ms) median {statistics.median(dec):.1f}")
    # time per step spent in main by reasoning bucket
    # seconds attributable to reasoning ~ rtok/tok_s
