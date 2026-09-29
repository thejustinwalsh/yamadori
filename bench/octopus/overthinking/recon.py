import json,sys,os,re,hashlib
from common import hermes, hm
sys.stdout.reconfigure(encoding='utf-8')
ROOTS={'v0b':'/root/space-shooter/','v0e1':'/workspace/space-shooter/','v0e2':'/workspace/space-shooter/',
       'v0f1':'/workspace/space-shooter/','v0f2':'/workspace/space-shooter/'}
def events(key):
    """ordered (ts, kind, input, result, ok) for writes/patches/rm in the project"""
    H=hermes(key); pend=[]; out=[]
    for r in H:
        if r.get('type')=='tool_use': pend.append(r)
        elif r.get('type')=='tool_result':
            # match first pending of same name
            for j,u in enumerate(pend):
                if u['name']==r['name']:
                    pend.pop(j); break
            else:
                continue
            out.append((u['timestamp']/1000, u['name'], u['input'], r.get('output',''), not r.get('is_error')))
    return out
def fuzzy_replace(text, old, new, replace_all=False):
    if old in text:
        return (text.replace(old,new) if replace_all else text.replace(old,new,1)), 'exact'
    # line-trimmed match
    tl=text.split('\n'); ol=old.split('\n')
    st=[l.strip() for l in tl]; so=[l.strip() for l in ol]
    for i in range(len(tl)-len(ol)+1):
        if st[i:i+len(ol)]==so:
            return '\n'.join(tl[:i]+new.split('\n')+tl[i+len(ol):]), 'trimmed'
    # whitespace-collapsed
    return None, 'nomatch'
def apply_result_diff(text, res):
    """Apply the unified diff Hermes returned (its own fuzzy match's result)."""
    try:
        d=json.loads(res).get('diff')
    except Exception:
        return None
    if not d: return None
    lines=text.split('\n')
    hunks=re.split(r'\n(?=@@ )', d)
    offset=0
    for hk in hunks:
        m=re.match(r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@', hk)
        if not m: continue
        body=hk.split('\n')[1:]
        old=[l[1:] for l in body if l.startswith(' ') or l.startswith('-')]
        new=[l[1:] for l in body if l.startswith(' ') or l.startswith('+')]
        start=int(m.group(1))-1+offset
        if lines[start:start+len(old)]!=old:
            # search nearby
            found=None
            for s in range(max(0,start-50),min(len(lines),start+50)):
                if lines[s:s+len(old)]==old: found=s; break
            if found is None: return None
            start=found
        lines[start:start+len(old)]=new
        offset+=len(new)-len(old)
    return '\n'.join(lines)
def replay(key, state=None, stop_at=None, log=None):
    state=dict(state or {})
    root=ROOTS[key]; ev=[]
    for ts,name,inp,res,ok in events(key):
        if stop_at and ts>stop_at: break
        p=inp.get('path','') if isinstance(inp,dict) else ''
        if name=='write_file' and p.startswith(root) and ok:
            state[p[len(root):]]=inp.get('content','')
            ev.append((ts,'write',p[len(root):],len(inp.get('content',''))))
        elif name=='patch' and p.startswith(root):
            rel=p[len(root):]
            if not ok or '"success": false' in res[:100]:
                ev.append((ts,'patch-failed',rel,0)); continue
            if rel not in state:
                ev.append((ts,'patch-unknown-file',rel,0)); continue
            nt,how=fuzzy_replace(state[rel],inp.get('old_string',''),inp.get('new_string',''),inp.get('replace_all',False))
            if how!='exact':
                nd=apply_result_diff(state[rel],res)
                if nd is not None: nt,how=nd,'resultdiff'
            if nt is None:
                ev.append((ts,'patch-NOMATCH',rel,0)); continue
            state[rel]=nt; ev.append((ts,'patch-'+how,rel,len(inp.get('new_string',''))-len(inp.get('old_string',''))))
        elif name=='terminal':
            # `rm [-flags] FILE...` of project files: the command's own `cd ROOT`
            # makes relative names project-relative; an absolute name under ROOT
            # counts too (grader v2 regrade, 2026-09-26: v0f's scratch
            # .test-*.js files were removed this way; before, only
            # js/.verify-grid.js was handled). Globs match the state's keys.
            c=inp.get('command','')
            in_root=('cd '+root.rstrip('/')) in c
            for m in re.finditer(r'(?:^|[;&|\n(]\s*)rm\s+((?:-\w+\s+)*)([^;&|\n)]+)',c):
                for tok in m.group(2).split():
                    tok=tok.strip('\'"')
                    if tok.startswith('>') or tok.startswith('2>'): break
                    if tok.startswith(root): rel=tok[len(root):]
                    elif tok.startswith('/') or not in_root: continue
                    else: rel=tok[2:] if tok.startswith('./') else tok
                    import fnmatch
                    for k in [k for k in state if fnmatch.fnmatchcase(k,rel)]:
                        del state[k]; ev.append((ts,'rm',k,0))
    return state, ev
def write_state(state, d):
    import shutil
    if os.path.isdir(d): shutil.rmtree(d)
    for rel,txt in state.items():
        fp=os.path.join(d,rel); os.makedirs(os.path.dirname(fp),exist_ok=True)
        open(fp,'w',encoding='utf-8',newline='').write(txt)
def h(s): return hashlib.sha1(s.encode('utf-8')).hexdigest()[:10]
if __name__=='__main__':
    for key in ('v0b','v0e1'):
        st,ev=replay(key)
        print('==',key,len(ev),'events')
        import collections
        print(collections.Counter(e[1] for e in ev))
        for e in ev:
            if 'NOMATCH' in e[1] or 'unknown' in e[1] or 'trimmed' in e[1]: print('  ',hm(e[0]),e)
        if key=='v0e1':
            st2,ev2=replay('v0e2',state=st)
            print('== v0e2',collections.Counter(e[1] for e in ev2)); 
            for e in ev2: print('  ',hm(e[0]),e)
            actual=r'C:\Users\jwals\octo\runs\v0e-V0-xhigh-1\space-shooter'; final=st2
        else:
            actual=r'C:\Users\jwals\octo\logs\v0b-V0-xhigh-1\root_mirror\space-shooter'; final=st
        for dp,dn,fn in os.walk(actual):
            for f in fn:
                rel=os.path.relpath(os.path.join(dp,f),actual).replace(os.sep,'/')
                a=open(os.path.join(dp,f),encoding='utf-8',newline='').read()
                r=final.get(rel)
                print('   ',rel,'MATCH' if r==a else ('MISSING' if r is None else f'DIFF len {len(a)} vs {len(r)}'))
        print('   recon-only:',[k for k in final if not os.path.exists(os.path.join(actual,k))])
