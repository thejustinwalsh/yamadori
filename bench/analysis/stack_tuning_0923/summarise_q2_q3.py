"""Paired summaries for Q2 (code quality, runtime) and Q3 (per-class arm
results with the fan-out/repair record). Reads data/code_quality.json,
data/split_detector.json and the run's rows; no service calls."""
import json, os, statistics as st
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
RUN = os.path.join(ROOT, "bench", "livebench", "results", "lb-20260923-minp0")


def sign_p(pos, neg):
    n = pos + neg
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(pos, neg) + 1)) / 2 ** n)


cq = json.load(open(os.path.join(HERE, "data", "code_quality.json")))
by = {(r["arm"], r["id"]): r for r in cq}
ids = sorted({r["id"] for r in cq})
det = {x["id"]: x for x in json.load(open(os.path.join(HERE, "data", "split_detector.json"))) if x["set"] == "lb"}
sens = {(r["arm"], r["id"]): r["with_stdin_buffer"] for r in json.load(open(os.path.join(RUN, "stdin_buffer_sensitivity.json")))["rows"]}
rows = {}
for arm, fn in (("bonsai", "rows_bonsai_coding.jsonl"), ("yamadori-xhigh", "rows_yamadori-xhigh_coding.jsonl")):
    for line in open(os.path.join(RUN, fn)):
        r = json.loads(line); rows[(arm, r["id"])] = r


def metric(r, m):
    if m == "ruff_total": return r["ruff"]["total"]
    if m == "ruff_idiom": return r["ruff"]["idiom"]
    if m == "ruff_E_F": return sum(v for k, v in r["ruff"]["families"].items() if k in ("E", "F"))
    if m == "mccabe_max": return r["ruff"]["mccabe_max"]
    return r[m]


print("=== Q2: paired per question (yamadori-xhigh minus bonsai)")
for cls in ("all", "LCB_generation", "coding_completion"):
    sub = [i for i in ids if cls == "all" or by[("bonsai", i)]["task"] == cls]
    print(f"\n-- {cls} (n={len(sub)})")
    for m in ("sloc", "chars", "ast_cc", "mccabe_max", "ruff_total", "ruff_E_F", "ruff_idiom"):
        pairs = [(metric(by[("bonsai", i)], m), metric(by[("yamadori-xhigh", i)], m)) for i in sub]
        pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
        diffs = [b - a for a, b in pairs]
        up = sum(1 for d in diffs if d > 0); dn = sum(1 for d in diffs if d < 0)
        print(f"  {m:11} bonsai median {st.median(a for a, _ in pairs):>6}  xhigh median {st.median(b for _, b in pairs):>6}  "
              f"sum {sum(a for a, _ in pairs):>5} vs {sum(b for _, b in pairs):>5}  xhigh higher {up}, lower {dn}, same {len(diffs) - up - dn}  sign p={sign_p(up, dn):.3f}  (n={len(pairs)})")

print("\n-- ruff codes by arm (all 21)")
for arm in ("bonsai", "yamadori-xhigh"):
    codes = {}
    for i in ids:
        for k, v in by[(arm, i)]["ruff"]["codes"].items():
            codes[k] = codes.get(k, 0) + v
    print(f"  {arm:15}", sorted(codes.items(), key=lambda kv: -kv[1])[:18])
print("\n-- rebuilt program fails to parse (ast)")
for arm in ("bonsai", "yamadori-xhigh"):
    print(f"  {arm:15}", [i[:8] for i in ids if by[(arm, i)]["ast_cc"] is None])
print("\n-- uses stdin/stdout .buffer")
for arm in ("bonsai", "yamadori-xhigh"):
    print(f"  {arm:15}", [i[:8] for i in ids if by[(arm, i)]["buffer"]])

print("\n=== Q2 runtime: sum of per-test min time, private tests, questions where BOTH arms pass every private test")
both = [i for i in ids if all(all(t.get("ok") for t in by[(a, i)]["private"]) and by[(a, i)]["private"] for a in ("bonsai", "yamadori-xhigh"))]
ratios = []
for i in both:
    tb = sum(t["s"] for t in by[("bonsai", i)]["private"]); tx = sum(t["s"] for t in by[("yamadori-xhigh", i)]["private"])
    pb = sum(t["s"] for t in by[("bonsai", i)]["public"]); px = sum(t["s"] for t in by[("yamadori-xhigh", i)]["public"])
    ratios.append(tx / tb if tb else None)
    print(f"  {i[:8]} {by[('bonsai', i)]['task'][:10]:10} {by[('bonsai', i)]['testtype']:10} private bonsai {tb*1000:9.2f} ms  xhigh {tx*1000:9.2f} ms  ratio {tx/tb if tb else float('nan'):6.2f}   public {pb*1e6:8.1f} us vs {px*1e6:8.1f} us")
rs = [r for r in ratios if r]
faster = sum(1 for r in rs if r < 0.9); slower = sum(1 for r in rs if r > 1.1)
print(f"  n={len(rs)} geometric-mean ratio {st.geometric_mean(rs):.2f}; xhigh >10% faster {faster}, >10% slower {slower}, within 10% {len(rs)-faster-slower}; sign p={sign_p(faster, slower):.3f}")

print("\n=== Q3: per class (true label; the detector R_text agrees on all 21)")
for cls in ("LCB_generation", "coding_completion"):
    sub = [i for i in ids if by[("bonsai", i)]["task"] == cls]
    for lab, get in (("official", lambda a, i: rows[(a, i)]["score"]),
                     (".buffer provided", lambda a, i: sens.get((a, i), rows[(a, i)]["score"]))):
        b = [get("bonsai", i) for i in sub]; x = [get("yamadori-xhigh", i) for i in sub]
        xo = sum(1 for p, q in zip(b, x) if q > p); bo = sum(1 for p, q in zip(b, x) if p > q)
        print(f"  {cls:18} {lab:17} n={len(sub)} bonsai {sum(b)} xhigh {sum(x)}  xhigh-only {xo} bonsai-only {bo}  exact McNemar p={sign_p(xo, bo):.3f}")
print("\n-- each discordant question")
for i in ids:
    b, x = rows[("bonsai", i)]["score"], rows[("yamadori-xhigh", i)]["score"]
    if b == x:
        continue
    m = rows[("yamadori-xhigh", i)]["mechanisms"]
    fo, rp = m["fanout"], m["repair"]
    print(f"  {i[:8]} {by[('bonsai', i)]['task'][:10]:10} bonsai {b} xhigh {x} | detector R_text={det[i[:8]]['R_text']} | fanout steps {fo.get('steps')} method {fo.get('method')} winner {fo.get('winner')} replaced {fo.get('replaced')} parse {fo.get('candidates_parse')} | repair rounds {rp.get('rounds')} {rp.get('stopped')} | buffer x={by[('yamadori-xhigh', i)]['buffer']} b={by[('bonsai', i)]['buffer']} | bonsai parses={by[('bonsai', i)]['ast_cc'] is not None} xhigh parses={by[('yamadori-xhigh', i)]['ast_cc'] is not None}")
print("\n-- fan-out record by class (xhigh)")
for cls in ("LCB_generation", "coding_completion"):
    sub = [i for i in ids if by[("bonsai", i)]["task"] == cls]
    fo = [rows[("yamadori-xhigh", i)]["mechanisms"]["fanout"] for i in sub]
    rp = [rows[("yamadori-xhigh", i)]["mechanisms"]["repair"] for i in sub]
    print(f"  {cls}: n={len(sub)} replaced {sum(1 for f in fo if f.get('replaced'))}, all candidates parse {sum(1 for f in fo if f.get('candidates_parse') and all(f['candidates_parse']))}, none parse {sum(1 for f in fo if f.get('candidates_parse') and not any(f['candidates_parse']))}, methods {sorted((f.get('method') or '-') for f in fo)}, repair ran {sum(1 for r in rp if r.get('rounds'))}")
    tok_b = [rows[("bonsai", i)]["tokens"]["output"] for i in sub]; tok_x = [rows[("yamadori-xhigh", i)]["tokens"]["output"] for i in sub]
    sec_b = [rows[("bonsai", i)]["seconds"] for i in sub]; sec_x = [rows[("yamadori-xhigh", i)]["seconds"] for i in sub]
    print(f"     output tokens median bonsai {st.median(tok_b)} xhigh {st.median(tok_x)}; seconds median {st.median(sec_b):.0f} vs {st.median(sec_x):.0f} (timing epochs differ; see condition.json)")
