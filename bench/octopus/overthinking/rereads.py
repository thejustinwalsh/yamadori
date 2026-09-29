import bisect
from classify import R, ROOTS, PRODUCT, events
from recon import replay, fuzzy_replace, apply_result_diff
def comp_times(key): return [r['t0'] for r in R[key] if r['compaction']]
def rereads(key, init_state=None):
    root=ROOTS[key]; state=dict(init_state or {}); last_read={}; out=[]
    ct=comp_times(key); rows=R[key]; t0s=[r['t0'] for r in rows]
    for ts,name,inp,res,ok in events(key):
        p=(inp.get('path') or '') if isinstance(inp,dict) else ''
        rel=p[len(root):] if p.startswith(root) else None
        if name=='write_file' and rel and ok: state[rel]=inp.get('content','')
        elif name=='patch' and rel and ok and '"success": false' not in res[:100] and rel in state:
            nt,how=fuzzy_replace(state[rel],inp.get('old_string',''),inp.get('new_string',''))
            if how!='exact':
                nd=apply_result_diff(state[rel],res)
                nt = nd if nd is not None else nt
            if nt is not None: state[rel]=nt
        elif name=='read_file' and rel and PRODUCT.search(rel):
            cur=hash(state.get(rel,''))
            prev=last_read.get(rel)
            i=bisect.bisect_right(t0s,ts)-1
            if prev is not None and prev[1]==cur:
                comp_between=any(prev[0]<c<ts for c in ct)
                out.append(dict(ts=ts,rel=rel,req=rows[i]['i'],comp_between=comp_between,same_range=(prev[2]==(inp.get('offset'),inp.get('limit'))),chars=len(res)))
            last_read[rel]=(ts,cur,(inp.get('offset'),inp.get('limit')))
    return out, state
if __name__=='__main__':
    tot={}
    for key in ('v0b','v0e1','v0e2'):
        init=None
        if key=='v0e2': init,_=replay('v0e1')
        rr,_=rereads(key,init)
        nreads=sum(1 for ts,n,i,res,ok in events(key) if n=='read_file' and PRODUCT.search((i.get('path') or '')))
        rows=R[key]; t0s=[r['t0'] for r in rows]
        # cost: sum of dur of requests that were answered by the reread? attribute the NEXT request (which processed the reread result) -- use request that issued it
        issued=set(x['req'] for x in rr)
        cost=sum(r['dur']-((r['inv_s'] or 0) if r['deep_fire'] else 0) for r in rows if r['i'] in issued)
        print(key,'product reads',nreads,'unchanged re-reads',len(rr),'after a compaction',sum(x['comp_between'] for x in rr),'same range',sum(x['same_range'] for x in rr),'result chars',sum(x['chars'] for x in rr),'issuing-request time %.0f min'%(cost/60))
