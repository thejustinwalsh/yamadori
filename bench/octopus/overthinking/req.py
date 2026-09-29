import sys, json, collections
from common import RUNS, OUT, hm, posts
sys.stdout.reconfigure(encoding='utf-8')
out = {}
for key in RUNS:
    P = posts(key)
    rows = []
    for i, r in enumerate(P, 1):
        rs = r.get('response') or {}
        x = rs.get('x_yamadori') or {}
        u = rs.get('usage') or {}
        cd = (u.get('completion_tokens_details') or {})
        dp = x.get('deep') or {}
        inv = x.get('investigate') or {}
        tc = x.get('tool_code') or {}
        fx = tc.get('fixup') or {}
        fo = x.get('fanout') or {}
        c = x.get('cache') or {}
        b = x.get('budget') or {}
        rows.append(dict(
            i=i, t0=r['t0'], t_end=r.get('t_end', r['t0']), dur=round(r.get('t_end', r['t0']) - r['t0'], 1),
            status=r.get('status'), finish=rs.get('finish_reason'),
            prompt=u.get('prompt_tokens'), completion=u.get('completion_tokens'),
            reasoning_tok=cd.get('reasoning_tokens'), reasoning_chars=rs.get('reasoning_chars'),
            content_chars=rs.get('content_chars'), tool_calls=rs.get('tool_calls'),
            route=(x.get('route') or {}).get('class'), utility=x.get('utility'), utility_kind=x.get('utility_kind'),
            compaction=bool(x.get('compaction')),
            deep_fire=dp.get('fire'), deep_kind=dp.get('kind'), deep_job=dp.get('job'),
            deep_because=(dp.get('because') or '')[:200],
            think_calls=len(((dp.get('think_tool') or {}).get('calls')) or []),
            inv_ran=inv.get('ran'), inv_s=inv.get('seconds'), inv_searches=inv.get('searches'),
            inv_handoff=(inv.get('handoff') or {}).get('chars'),
            fanout=bool(fo), fanout_s=fo.get('seconds') if isinstance(fo, dict) else None,
            tc_rounds=tc.get('rounds'), tc_stopped=tc.get('stopped'), tc_fixup=bool(fx),
            model_ms=c.get('model_ms'), cache_reused=c.get('reused'), cache_processed=c.get('processed'),
            gens=c.get('generations'),
            budget_think=b.get('reasoning_budget_tokens'), nudge_at=b.get('nudge_at'),
            hops=x.get('hops'), last_role=(r.get('request') or {}).get('last_role'),
            n_messages=(r.get('request') or {}).get('n_messages'),
            last_head=((r.get('request') or {}).get('last_head') or '')[:160],
            err=(rs.get('error') or {}).get('message', '')[:160] if isinstance(rs.get('error'), dict) else rs.get('error'),
            per_gen=(x.get('usage') or {}).get('per_generation'),
        ))
    out[key] = rows
json.dump(out, open(OUT('requests.json'), 'w'), indent=0)
for key, rows in out.items():
    if not rows: print(key, 'no rows'); continue
    t0 = rows[0]['t0']; t1 = max(r['t_end'] for r in rows)
    comp = sum(r['completion'] or 0 for r in rows)
    rt = sum(r['reasoning_tok'] or 0 for r in rows)
    rc = sum(r['reasoning_chars'] or 0 for r in rows)
    cc = sum(r['content_chars'] or 0 for r in rows)
    print(f"== {key}: {len(rows)} requests, {hm(t0)}-{hm(t1)} = {(t1-t0)/3600:.2f} h; completion {comp}; reasoning_tok {rt}; reasoning_chars {rc}; content_chars {cc}")
    print('   status', collections.Counter(r['status'] for r in rows), 'finish', collections.Counter(r['finish'] for r in rows))
    print('   routes', collections.Counter(r['route'] for r in rows), 'utility', sum(1 for r in rows if r['utility']), 'compactions', sum(1 for r in rows if r['compaction']))
    print('   deep fired', [(r['i'], r['deep_kind'], r['inv_s'], r['inv_searches'], r['inv_handoff']) for r in rows if r['deep_fire']])
    print('   think_deeply calls', [(r['i'], r['think_calls']) for r in rows if r['think_calls']])
    print('   fixups', [(r['i'], r['tc_rounds'], r['tc_stopped']) for r in rows if r['tc_fixup']])
    print('   fanout', [r['i'] for r in rows if r['fanout']])
    print('   budget_think', collections.Counter(r['budget_think'] for r in rows))
    tc = collections.Counter(n for r in rows for n in (r['tool_calls'] or []))
    print('   tool calls', tc)
