"""Score the only stack-on/stack-off pairs the domain suite has in a language
its style scorer covers: smoke-medium A0 vs A5 on ts01, ts02, tg01, tg02.
bench/domain/grade_style.style() on the stored answer content; CPU only."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
DOM = os.path.abspath(os.path.join(HERE, "..", "..", "domain"))
sys.path.insert(0, DOM)
import grade as G  # noqa: E402
import grade_style  # noqa: E402
tasks = {t["id"]: t for t in G.load_tasks()}
rows = {}
for l in open(os.path.join(DOM, "results", "smoke-medium", "rows.jsonl"), encoding="utf-8"):
    if l.strip():
        r = json.loads(l); rows[(r["task"], r["arm"])] = r
out = []
for tid in ("ts01", "ts02", "tg01", "tg02"):
    for arm in ("A0", "A5"):
        a = json.load(open(os.path.join(DOM, "results", "smoke-medium", "answers", f"{tid}__{arm}__1.json"), encoding="utf-8"))
        s = grade_style.style(tasks[tid], a["content"])
        rec = {"task": tid, "arm": arm, "outcome": rows[(tid, arm)].get("outcome"),
               "lint_errors": s.get("lint_errors"), "lint_warnings": s.get("lint_warnings"),
               "rules": s.get("lint_rules"), "skipped": s.get("skipped"), "style_error": (s.get("style_error") or "")[:200],
               "hints": [(h.get("score"), h.get("source")) for h in ((rows[(tid, arm)].get("x_yamadori") or {}).get("hints") or [])]}
        out.append(rec); print(rec)
json.dump(out, open(os.path.join(HERE, "data", "style_smoke_pairs.json"), "w"), indent=1)
