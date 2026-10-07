"""Render RESULT-strat860.md from subset-summary.json (score_subset.py output).  python make_result_md.py SUMMARY.json OUT.md"""
import json, sys
d = json.load(open(sys.argv[1]))
f = lambda x: '-' if x is None else f'{x*100:.0f}'
out = []
out.append("""# jjava on the Decision Index 0.2.1 harness: 860-row subset (2026-10-07)

**What this is.** The harness's own `suite sample --n 860` (round-robin by benchmark, complete groups, seed 20260919) over a
locally rebuilt corpus, run through our `/jev` API (`jjava-latest` -> `jjava-bonsai-a4000`), one request at a time:
860 requests, 860 ok, 0 unsupported, 0 errors, 0 waits, 10,587 s (median request 2.7 s, p95 60.7 s, mean 12.3 s).

**Labels (all apply to every number below).**
- A SUBSET: about 17 scored cases per benchmark (n in the table), so every per-benchmark score is very noisy.
- **HLE is not built** (gated on the Hub): 37 of the 38 counted benchmarks are present; the corpus sha256 is not the frozen
  one (the 7 added benchmarks and the 42 other sources were verified byte-for-byte; see README.md).
- **No index is reported.** Not leaderboard-comparable.
- Scores are the harness's own per-benchmark chance-corrected skill (`index02`), computed COVERAGE-FREE on the rows run
  (`score_subset.py`; the harness's own `score` counts every unrun request as wrong and returns ~0.3 for a subset).
  Chance is the board's full-benchmark chance, not recomputed for the subset. Reference columns are the board's
  full-benchmark skill (`data/index-v0.2.1.json`, all rows, coverage 1).
- Intervals: Wilson 95% on the raw score as if a proportion, mapped through the chance correction. Crude for F1, nDCG,
  macro-F1; none for ForecastBench (Brier vs the 0.5 baseline).
- **"Clef-flash" is not an entrant on the 0.2.1 board** (no match for `clef` in the board file or the methodology). The
  columns are Jev (jev-1.13.0), Kev 9B (`kev-9b-raised`), and for context the top open entrant Surogate Rune 26B-A4B v3 and
  simple-jev on Qwen3.8-27B (the 27B family jjava's Bonsai 2 is built on).

Skill, 0-100 (0 = chance, 100 = perfect). `n` = scored cases (queries / reviews for ToolRet, BRIGHT, ACOS; requests otherwise).
Flag: Jev's value outside jjava's interval.

| id | benchmark | n | jjava | 95% interval | Jev | Kev 9B | Rune 26B | simple-jev 27B | flag vs Jev |
|---|---|---|---|---|---|---|---|---|---|""")
for r in d['rows']:
    j = r['jev']; flag = ''
    if j is not None and r['lo'] is not None:
        flag = 'BEHIND' if j > r['hi'] else 'ahead' if j < r['lo'] else ''
    nc = '' if r['in_index'] else ' (not counted)'
    iv = '-' if r['lo'] is None else f"{r['lo']*100:.0f}-{r['hi']*100:.0f}"
    out.append(f"| {r['id']} | {r['name']}{nc} | {r['n']} | {f(r['skill'])} | {iv} | {f(r['jev'])} | {f(r['kev-9b-raised'])} | {f(r['rune-26b-a4b-v3'])} | {f(r['simple-jev-qwen3.8-27b'])} | {flag} |")
a = {x['id']: x for x in d['areas_indicative']}
out.append("""
## Reading it

**Clearly behind Jev** (Jev's value is above jjava's interval; n = 17 each unless noted): CLINC150+OOS, cfcolor, GPQA Diamond,
WinoGrande, GSM8K (n 34), CRUXEval, CLadder, MMLU-Pro, BBH, and iSarcasmEval (but see below). The knowledge/reasoning
benchmarks are the pattern: MMLU-Pro 34 vs Jev 81, BBH 40 vs 90, GPQA 6 vs 71, CRUXEval 15 vs 57, CLadder 0 vs 45, WinoGrande
18 vs 84. Jev's lead there is large enough to survive the noise; the cause is not established (jjava reads one token per
option with thinking off; Jev's method is not public).

**Clearly ahead of Jev:** none by the interval rule. Nominally higher, with intervals that include Jev: ToolRet (71 vs 60),
API-Bank (94 vs 88), BANKING77 (92 vs 79, but see below), BPoMP (87 vs 82, n 55), MuSR (63 vs 46), RAGTruth (67 vs 51),
HoVer (53 vs 46), ContractNLI (67 vs 59). Against Kev 9B, jjava is nominally ahead on most of the same.

**In the noise** (interval contains Jev): BFCL, RouterBench (not counted), Home appliances, ANLI, Humicroedit, POP909, MMLU
(not counted), HellaSwag, ChessBench, SATA, BRIGHT, ESCI, ACOS, FinEntity, VAST, NLI4CT, Habermas, PhishNChips, When2Call,
New Yorker.

**Do not read as signal:**
- BANKING77 (77 classes) and CLINC150 (151) are macro-F1 over about 17 requests; the label universe is whatever was seen,
  so the value is not comparable to the board's. The 92 for BANKING77 is not a real lead; CLINC150's 59 vs 89 is not clean either.
- iSarcasmEval scores track A-English F1 on 4 rows: 0 is uninformative. ForecastBench has no interval (Brier above the
  baseline gives 0).
- Skill clips at 0, so several 0 entries (cfcolor, CLadder, SimpleBench n=10, Habermas) mean "at or below chance" at n=17.

**Pooled, indicative only.** Mean per-benchmark skill over the 37 counted benchmarks present: jjava 41.0; Jev 55.9
(difference -14.9; sd of per-benchmark differences 21.8, so about +-7 at 95% if the benchmarks were independent draws;
jjava ahead by more than 5 points on 7, behind by more than 5 on 24, within 5 on 6). Kev 9B 36.2 (jjava +4.8, +-7: not
separable). Rune 56.3, simple-jev 27B 53.8. This is NOT the board's index (weights, HLE, coverage differ) and must not be
quoted as one.

**Areas (the harness's weighted area means of the benchmarks present, indicative only):** knowledge @K@,
language @L@, retrieval & classification @R@, tools @T@, arts @A@.

## Reproduce

```
cd C:/Users/jwals/octo/decision-index ; $env:PYTHONUTF8=1
.venv/Scripts/python.exe C:/Users/jwals/llama-stack/bench/decider/decision_index/score_subset.py runs/jjava-strat860 --out runs/jjava-strat860/subset-summary.json
.venv/Scripts/python.exe C:/Users/jwals/llama-stack/bench/decider/decision_index/make_result_md.py runs/jjava-strat860/subset-summary.json RESULT-strat860.md
```
Result rows: `runs/jjava-strat860/results.jsonl` (outside the repo; not committed). Run: `run_pass.ps1 -Rows strat-860.jsonl.gz
-Out runs/jjava-strat860`. This is one run: repeat before citing a difference (docs/PROTOCOL.md).
""")
out[-1] = (out[-1].replace("@K@", f"{a['knowledge']['skill']*100:.0f}").replace("@L@", f"{a['language']['skill']*100:.0f}").replace("@R@", f"{a['retrieval']['skill']*100:.0f}").replace("@T@", f"{a['tools']['skill']*100:.0f}").replace("@A@", f"{a['arts']['skill']*100:.0f}"))
open(sys.argv[2], 'w', encoding='utf-8', newline='\n').write("\n".join(out) + "\n")
