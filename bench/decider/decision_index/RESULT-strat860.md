# jjava on the Decision Index 0.2.1 harness: 860-row subset (2026-10-07)

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
|---|---|---|---|---|---|---|---|---|---|
| 1 | BFCL | 18 | 93 | 68-99 | 94 | 93 | 93 | 96 |  |
| 2 | ToolRet | 18 | 71 | 44-87 | 60 | 59 | 59 | 63 |  |
| 3 | API-Bank | 18 | 94 | 74-99 | 88 | 55 | 83 | 85 |  |
| 4 | BANKING77 | 18 | 92 | 70-98 | 79 | 85 | 84 | 79 |  |
| 5 | CLINC150+OOS | 17 | 59 | 36-79 | 89 | 79 | 87 | 88 | BEHIND |
| 6 | RouterBench (not counted) | 34 | 38 | 0-66 | 53 | 53 | 53 | 52 |  |
| 9 | Home appliance simulator | 17 | 41 | 22-64 | 52 | 25 | 47 | 65 |  |
| 10 | SGD/SGD-X (not counted) | 17 | 65 | 26-87 | - | - | - | - |  |
| 11 | ContractNLI | 17 | 67 | 33-87 | 59 | 39 | 67 | 68 |  |
| 12 | ANLI | 17 | 39 | 5-68 | 62 | 34 | 60 | 54 |  |
| 20 | BPoMP | 55 | 87 | 67-95 | 82 | 34 | 80 | 93 |  |
| 21 | Humicroedit | 17 | 18 | 0-57 | 24 | 11 | 24 | 23 |  |
| 22 | POP909-CL | 17 | 17 | 5-41 | 16 | 9 | 66 | 28 |  |
| 23 | cfcolor | 17 | 0 | 0-10 | 29 | 11 | 25 | 27 | BEHIND |
| 24 | MMLU (not counted) | 17 | 84 | 54-96 | 89 | 68 | 79 | 79 |  |
| 25 | GPQA Diamond | 17 | 6 | 0-38 | 71 | 18 | 29 | 31 | BEHIND |
| 26 | ARC-Easy (not counted) | 17 | 92 | 64-99 | - | - | - | - |  |
| 27 | ARC-Challenge (not counted) | 17 | 92 | 64-99 | - | - | - | - |  |
| 28 | WinoGrande | 17 | 18 | 0-57 | 84 | 46 | 72 | 69 | BEHIND |
| 29 | HellaSwag | 17 | 84 | 54-96 | 93 | 76 | 90 | 91 |  |
| 30 | GSM8K | 34 | 51 | 23-65 | 76 | 38 | 76 | 58 | BEHIND |
| 31 | ChessBench | 17 | 9 | 0-35 | 10 | 3 | 12 | 10 |  |
| 32 | MuSR | 17 | 63 | 25-85 | 46 | 33 | 44 | 41 |  |
| 33 | SATA-Bench | 17 | 23 | 8-47 | 25 | 26 | 34 | 31 |  |
| 34 | SimpleBench (not counted) | 10 | 0 | 0-28 | - | - | - | - |  |
| 36 | BRIGHT | 17 | 33 | 12-59 | 41 | 30 | 39 | 43 |  |
| 37 | Amazon ESCI | 17 | 32 | 6-59 | 44 | 36 | 44 | 47 |  |
| 38 | ACOS | 17 | 17 | 4-42 | 27 | 16 | 24 | 34 |  |
| 39 | FinEntity | 17 | 73 | 38-90 | 81 | 83 | 83 | 83 |  |
| 40 | iSarcasmEval | 17 | 0 | 0-0 | 36 | 32 | 49 | 49 | BEHIND |
| 41 | VAST | 17 | 29 | 0-61 | 47 | 33 | 65 | 58 |  |
| 42 | NLI4CT | 17 | 61 | 15-85 | 69 | 51 | 62 | 68 |  |
| 43 | CRUXEval | 17 | 15 | 0-50 | 57 | 23 | 59 | 40 | BEHIND |
| 44 | CLadder | 17 | 0 | 0-28 | 45 | 24 | 45 | 42 | BEHIND |
| 48 | ForecastBench | 17 | 0 | - | 31 | 30 | 19 | 15 |  |
| 50 | Habermas Machine | 17 | 0 | 0-34 | 22 | 12 | 16 | 14 |  |
| 56 | PhishNChips phishing decisions | 17 | 29 | 0-65 | 25 | 2 | 62 | 69 |  |
| 57 | MMLU-Pro | 17 | 34 | 12-60 | 81 | 45 | 63 | 55 | BEHIND |
| 58 | BBH fixed-option tasks | 17 | 40 | 7-69 | 90 | 50 | 73 | 58 | BEHIND |
| 59 | RAGTruth response-level hallucination | 17 | 67 | 19-89 | 51 | 0 | 52 | 43 |  |
| 61 | HoVer claim verification | 17 | 53 | 5-81 | 46 | 18 | 61 | 49 |  |
| 62 | When2Call MCQ | 17 | 53 | 22-77 | 75 | 33 | 68 | 67 |  |
| 64 | New Yorker caption matching | 17 | 49 | 20-73 | 63 | 48 | 68 | 61 |  |

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

**Areas (the harness's weighted area means of the benchmarks present, indicative only):** knowledge 24,
language 46, retrieval & classification 51, tools 72, arts 24.

## Reproduce

```
cd C:/Users/jwals/octo/decision-index ; $env:PYTHONUTF8=1
.venv/Scripts/python.exe C:/Users/jwals/llama-stack/bench/decider/decision_index/score_subset.py runs/jjava-strat860 --out runs/jjava-strat860/subset-summary.json
.venv/Scripts/python.exe C:/Users/jwals/llama-stack/bench/decider/decision_index/make_result_md.py runs/jjava-strat860/subset-summary.json RESULT-strat860.md
```
Result rows: `runs/jjava-strat860/results.jsonl` (outside the repo; not committed). Run: `run_pass.ps1 -Rows strat-860.jsonl.gz
-Out runs/jjava-strat860`. This is one run: repeat before citing a difference (docs/PROTOCOL.md).

