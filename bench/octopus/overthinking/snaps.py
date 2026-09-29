import sys, os, datetime
from recon import replay, write_state
from common import OUT_DIR
SCR = OUT_DIR
sys.path.insert(0, r'C:\Users\jwals\llama-stack\bench\octopus')
import grade as G
G.GRADES = os.path.join(SCR, 'grades_snap.jsonl')   # never the repo's results file
def at(key, hhmmss, day):
    return datetime.datetime.strptime(f'{day} {hhmmss}', '%Y-%m-%d %H:%M:%S').timestamp() + 0.999
plan = []
# v0b: product states after each product edit in the first two hours
for t, d in [('00:11:20','2026-09-25'),('00:13:52','2026-09-25'),('00:17:50','2026-09-25'),('00:22:27','2026-09-25')]:
    st,_ = replay('v0b', stop_at=at('v0b', t, d))
    plan.append((f"v0b-at-{t.replace(':','')}", st))
# v0e p1: counterfactual = JS as of T + the entry files the model wrote at the end
final,_ = replay('v0e1')
entry = {k: final[k] for k in ('index.html','css/styles.css','README.md')}
for t in ['19:58:51','20:11:24','20:43:52','21:40:30','22:07:06']:
    st,_ = replay('v0e1', stop_at=at('v0e1', t, '2026-09-25'))
    st = {k:v for k,v in st.items() if not k.startswith('js/.verify')}
    st.update(entry)
    plan.append((f"v0e1-cf-{t.replace(':','')}", st))
for t in ['22:16:26','22:17:08']:
    st,_ = replay('v0e1', stop_at=at('v0e1', t, '2026-09-25'))
    plan.append((f"v0e1-at-{t.replace(':','')}", st))
only = sys.argv[1:] 
for gid, st in plan:
    if only and gid not in only: continue
    d = os.path.join(SCR, 'snaps', gid)
    write_state(st, d)
    row = G.grade(gid, 'V0', d, None, True)
    print(gid, row['spec_passed'], row['spec_failed'], row['spec_unknown'],
          [c['check'] for c in row.get('spec') or [] if c.get('pass') is False], flush=True)
