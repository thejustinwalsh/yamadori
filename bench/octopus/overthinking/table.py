import sys,json
from classify import R, CPT, ROOTS, assign, cat_call
from common import hm
key=sys.argv[1]
rows=R[key]; per=assign(key); root=ROOTS[key]
for idx,r in enumerate(rows):
    calls=per.get(idx,[])
    desc=[]
    for ts,n,i,res,ok in calls:
        a=(i.get('path') or i.get('command') or i.get('pattern') or i.get('image_url','')[:30] or json.dumps(i)[:60]) if isinstance(i,dict) else ''
        a=a.replace(root,'').replace('\n',' ')[:70]
        desc.append(f"{n}:{cat_call(n,i,root)}:{a}{'' if ok else ' !ERR'}")
    rt=r['reasoning_tok'] or round((r['reasoning_chars'] or 0)/CPT)
    extra=''
    if r['deep_fire']: extra+=f" DEEP[{r['deep_kind']} {r['inv_s']}s srch{r['inv_searches']}]"
    if r['tc_fixup']: extra+=f" FIXUP[{r['tc_rounds']} {r['tc_stopped']}]"
    if r['compaction']: extra+=' COMPACTION'
    if r['utility']: extra+=' UTIL'
    if r['status']!=200: extra+=f" STATUS{r['status']}"
    print(f"{r['i']:3d} {hm(r['t0'])} {r['dur']:6.0f}s p{r['prompt'] or 0:6d} c{r['completion'] or 0:5d} r{rt:5d}{extra} | {' ; '.join(desc)[:260]}")
