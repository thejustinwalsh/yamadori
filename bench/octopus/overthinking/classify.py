import json,sys,re,collections,bisect
from common import OUT
from recon import events, ROOTS
sys.stdout.reconfigure(encoding='utf-8')
R=json.load(open(OUT('requests.json')))
CPT=3.88
PRODUCT=re.compile(r'(index\.html|css/styles\.css|README\.md|js/(config|audio|particles|background|enemies|player|ui|game)\.js)$')
def cat_call(name, inp, root):
    p=(inp.get('path') or '') if isinstance(inp,dict) else ''
    rel=p[len(root):] if p.startswith(root) else p
    c=(inp.get('command') or '') if isinstance(inp,dict) else ''
    if name in ('write_file','patch'):
        return 'product_edit' if PRODUCT.search(rel) and p.startswith(root) else 'harness_build'
    if name=='read_file':
        return 'read_product' if PRODUCT.search(rel) else 'read_other'
    if name=='search_files': return 'search'
    if name=='vision_analyze': return 'env_vision'
    if name=='terminal':
        if re.search(r'playwright|chromium|apt-get|apt-cache|ldd |headless|base64|screens|npm (i|install)|which chromium|ms-playwright|data:image|diag2?\.js|snap\.js|mkurl|genurl|fullurl|b64',c): return 'env_browser'
        if re.search(r'harness\.js|runtime_test|\.dbg|test_full|node -e|tmp_octo|vm\.create',c): return 'harness_run'
        if re.search(r'node --check|ls |find |wc |cat |sed -n|head |tail |grep |curl |http\.server|pwd|whoami|echo',c): return 'inspect_verify'
        return 'terminal_other'
    return 'other:'+name
def assign(key):
    rows=R[key]; t0s=[r['t0'] for r in rows]
    per=collections.defaultdict(list)
    for ts,name,inp,res,ok in events(key):
        i=bisect.bisect_right(t0s,ts)-1
        per[i].append((ts,name,inp,res,ok))
    return per
def classify(key, t_from=None, t_to=None):
    rows=R[key]; per=assign(key); root=ROOTS.get(key,'/workspace/space-shooter/')
    agg=collections.defaultdict(lambda: collections.Counter())
    for idx,r in enumerate(rows):
        if t_from and r['t0']<t_from: continue
        if t_to and r['t0']>=t_to: continue
        calls=per.get(idx,[])
        cats=[cat_call(n,i,root) for _,n,i,_,_ in calls]
        if r['compaction']: c='compaction'
        elif r['utility']: c='utility'
        elif r['status']!=200: c='failed_request'
        elif not calls: c='answer_or_empty'
        else:
            # precedence
            for pc in ('product_edit','harness_build','env_browser','env_vision','harness_run','read_product','search','inspect_verify','read_other','terminal_other'):
                if pc in cats: c=pc; break
            else: c=cats[0]
        deep=(r['inv_s'] or 0) if r['deep_fire'] else 0
        a=agg[c]
        a['n']+=1; a['s']+=r['dur']-deep; a['completion']+=r['completion'] or 0
        rt=r['reasoning_tok'] or (0 if (r['deep_fire'] and r['inv_s']) else round((r['reasoning_chars'] or 0)/CPT))
        a['reasoning']+=rt
        if deep:
            agg['deep_thinking']['n']+=1; agg['deep_thinking']['s']+=deep
        if r['tc_fixup']:
            fx=max(0,r['dur']-deep-(r['model_ms'] or 0)/1000)
            agg['fixup']['n']+=1; agg['fixup']['s']+=fx; a['s']-=fx
    return agg
def show(key,t_from=None,t_to=None,label=''):
    agg=classify(key,t_from,t_to)
    tot=sum(v['s'] for v in agg.values())
    print(f'-- {key} {label}: {tot/3600:.2f} h in requests')
    for c,v in sorted(agg.items(), key=lambda kv:-kv[1]['s']):
        print(f"   {c:16s} n={v['n']:3d} {v['s']/60:7.1f} min ({100*v['s']/max(tot,1):4.1f}%)  completion {v['completion']:6d} reasoning {v['reasoning']:6d}")
if __name__=='__main__':
    for k in ('v0b','v0e1','v0e2'): show(k,label='whole')
