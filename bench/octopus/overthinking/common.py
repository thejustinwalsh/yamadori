"""Overthinking analysis of the Octopus runs (docs/research/OVERTHINKING.md).

Offline and read-only: reads C:/Users/jwals/octo/logs/<run>/{relay,hermes}*.jsonl,
never a model. Run from this directory with the stack interpreter, in order:

    python req.py          # per-request table from the relay rows -> out/requests.json
    python timeline.py     # time accounting, thinking vs the caps
    python classify.py     # where each run's request time went, by what the step did
    python table.py v0b    # one line per request (v0b | v0e1 | v0e2)
    python recon.py        # rebuilds the project from write_file/patch calls; checks the
                           # rebuilt final state byte for byte against the real files
    python rereads.py      # product files re-read while unchanged
    python snaps.py        # grades rebuilt snapshots with grade.py (Docker, no GPU);
                           # rows go to out/grades_snap.jsonl, never results/grades.jsonl
"""
import json, os, datetime
LOGS = r"C:\Users\jwals\octo\logs"
RUNS = {
  'pilot': ('pilot-V0-xhigh-1', ''),
  'v0b':   ('v0b-V0-xhigh-1', ''),
  'v0c':   ('v0c-V0-xhigh-1', ''),
  'v0d':   ('v0d-V0-xhigh-1', ''),
  'v0e1':  ('v0e-V0-xhigh-1', ''),
  'v0e2':  ('v0e-V0-xhigh-1', '.p2'),
  'v0f1':  ('v0f-V0-xhigh-1', ''),
  'v0f2':  ('v0f-V0-xhigh-1', '.p2'),
}
def jl(p):
    out = []
    if not os.path.exists(p): return out
    for l in open(p, encoding='utf-8'):
        if not l.strip(): continue
        try: out.append(json.loads(l))
        except ValueError: out.append({'type': '_line', 'text': l.strip()})
    return out
def relay(key):
    rid, sfx = RUNS[key]
    return jl(os.path.join(LOGS, rid, f'relay{sfx}.jsonl'))
def hermes(key):
    rid, sfx = RUNS[key]
    return jl(os.path.join(LOGS, rid, f'hermes{sfx}.jsonl'))
def hm(t):
    return datetime.datetime.fromtimestamp(t).strftime('%H:%M:%S')
def posts(key):
    return [r for r in relay(key) if r.get('method') == 'POST' and r.get('path','').endswith('/chat/completions')]
# Every output of these scripts goes to out/ beside them (gitignored).
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out')
os.makedirs(OUT_DIR, exist_ok=True)
def OUT(name):
    return os.path.join(OUT_DIR, name)
