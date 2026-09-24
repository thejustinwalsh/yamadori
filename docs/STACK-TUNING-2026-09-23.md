# Stack tuning, 2026-09-23: hint rate, code quality, and where the stack loses

Written for the operator, who asked for three answers. Nothing in `mcp/` was
changed and nothing was committed. No chat model was called and nothing was
sent to `:1234`. The only service calls were **399 query embeddings** (303
for prompts, 89 for probes, 6 for the parity check and 1 liveness check) to
the embeddings model on llama-swap `:11434`, batched into 33 requests. Every script and its output
is under `bench/analysis/stack_tuning_0923/` (listed at the end).

Labels, as in `docs/DECISION-TREES-AND-SKILLS.md`: **[M]** measured here or in
a named artefact, with n; **[P]** proposed or untested. The embedder is
deterministic, so a re-run of an embedding measurement returns the same
numbers and is not a repeat in the PROTOCOL rule 10 sense. Every LiveBench
number below comes from **one run of n=21 at temperature 1.0**. The bare arm
disagreed with itself on 8 of those 21 questions between two runs (4 vs 4,
`lb-20260923-minp0/sanity.md`), so roughly 38% discordance is the noise
floor for any paired LiveBench comparison here.

---

## The answers

**Q1. Why is the hint rate so low?** Only one pipeline stage removes
anything, and it is the 0.55 similarity floor. The domain gate removed
nothing from any of the 141 prompts, because none of them carries an import
or a file extension, so no domain filter was ever applied. Bucket collapse
also removed nothing at 0.55. Beyond that, "rarely fires" is two separate
facts:

- **LiveBench: the corpus has nothing close to these prompts.** The best
  cosine per prompt ranges from 0.204 to 0.491 (median 0.336, n=21), so the
  floor could sit anywhere from 0.50 to 0.60 and LiveBench would still get
  0/21. Lowering the floor to 0.40 lets 5/21 through. By my reading, one of
  those five hints is apt.
- **Domain suite: the prompts do reach the corpus, but the arms that allow
  hints were barely run.** Offline, 59/120 uncontaminated prompts would get at
  least one hint at 0.55 (49%). The results hold only 12 hint-allowed rows,
  over 6 tasks, and A2 (hints only) never ran. Three of those six tasks sit
  below the floor offline, and the offline trace agrees with every silent
  production row.

The floor sweep changes selection, not quality. On the domain suite:

| floor | injected (of 120) |
|---|---|
| 0.45 | 112 |
| 0.50 | 94 |
| 0.55 | 59 |
| 0.60 | 32 |

On LiveBench the counts are 2, 0, 0 and 0. On the 89 labelled probes, moving
the floor from 0.55 to 0.50 shows the correct bucket member 44 times instead
of 27, and a wrong sibling 10 times instead of 9.

**Q2. Does the code look better with the stack?** Mostly no. On generation
the code is equivalent. On completion the stack's code compiles more often,
because it rewrites the whole function instead of continuing the starter.
Details:

- **Compile failures:** all 21 xhigh programs compile, against 18/21 for
  bonsai. All three bonsai failures are continuation-format errors on
  completion questions.
- **Lint findings the model is responsible for:** 67 vs 56, sign p = 1.0.
- **Complexity:** no difference (sign p ≥ 0.75).
- **Length:** completion answers are about 3× longer (sign p = 0.008),
  because 6/10 re-emit the whole `class Solution`.
- **Runtime:** on the 16 questions both arms pass, xhigh is more than 10%
  faster on 6 and more than 10% slower on 1 (sign p = 0.125). The geometric
  mean ratio is 0.68, driven by two questions.
- **Domain style score:** no stack-on arm has one yet. Scoring the only four
  A0 vs A5 TypeScript pairs offline gave A5 fewer lint errors on 2 and a tie
  on 2. That is n=4 on a stale stack, so it is a direction only.

**Q3. Where does the stack do worse, and can we route around it?** Measured
by LiveBench's own labels, the split is clean:

- **Generation (n=11):** the stack loses 0 vs 3 on official scores. All three
  losses are the `.buffer` grader quirk, and all three kept the main model's
  original. Once the grader provides `.buffer`, the arms tie at 10/11 each.
- **Completion (n=10):** the stack wins 3 vs 0. Two of the wins come from
  fan-out's tie-breaker rewriting the class, and one from a repair round.
  Exact McNemar p = 0.25.

So the stack did no measured harm on generation, but it also did no measured
good there, at a median of 6.8× the output tokens.

A detector that reads only the instruction text separates the two classes
128/128 on LiveBench and raises 0 false alarms on the 120 domain prompts.
Those 128 prompts use only two instruction templates, though, so that is two
distinct cases, not 128. The operator's literal rule ("the last block fails
to parse, or the text says continue, complete, finish or fix") scores 16/21
and 69/107, because generation stubs do not parse either.

Deciding "stack on for completion" at α = 0.025 (Bonferroni over the two
classes) needs about 50–100 completion questions. After removing every
problem already seen, LiveBench has 37 left.

---

## Q1. The hint pipeline, stage by stage

### Method [M]

`hint_trace.py` runs the production path offline:

- the served `index/hints.npz`: 1,860 rows, the signature matches the corpus,
  and there are no zero vectors;
- `hints.task_domains`, `domains.eligible` and the bucket collapse in
  `hints._select_vector`, with k = 3;
- one query embedding per prompt, through `code_search`'s embeddings call
  with the production `QUERY_INSTRUCT`.

It was checked against `hints.select()` itself on 6 prompts, and agreed 6/6.

Two prompt sets were traced:

- **LiveBench:** the 21 coding questions of `lb-20260923-minp0`, sent as
  LiveBench sends them (the whole of `turns[0]` as the user turn). With the
  code tools withheld, `augmented` is the client's own messages, so this is
  exactly what `hints.attach` saw.
- **Domain suite:** the 120 uncontaminated prompts (core without three_tsl,
  plus react), sent as `bench/domain/run.py` sends them.

### Where each prompt drops out, at the production floor 0.55 [M]

| set | n | domain gate applied | dropped at the floor | dropped at the domain gate | dropped at suppression | injected ≥1 |
|---|---|---|---|---|---|---|
| LiveBench, LCB_generation | 11 | 0 | 11 | 0 | 0 | 0 |
| LiveBench, coding_completion | 10 | 0 | 10 | 0 | 0 | 0 |
| domain, typescript | 30 | 0 | 18 | 0 | 0 | 12 |
| domain, typegpu | 25 | 0 | 6 | 0 | 0 | 19 |
| domain, rust_wasm | 25 | 0 | 19 | 0 | 0 | 6 |
| domain, react | 40 | 0 | 18 | 0 | 0 | 22 |
| **all** | **141** | **0** | **82** | **0** | **0** | **59** |

- **Domain gate.** `task_domains` returns `None`, meaning no filter, for all
  141 prompts. It only filters when a prompt carries an import or a file
  extension (`domains.strong_evidence`), and none does. The filter has never
  engaged on either benchmark.
- **Bucket suppression cannot silence a prompt.** The winner of a bucket is
  always kept. At 0.55 it withheld 0 rows. At 0.40 it withheld 3 rows on 1
  LiveBench prompt.
- **The floor is the only stage that drops anything.**

### The floor sweep: selection only [M]

On the traced prompts (`hint_report.py`):

| floor | LiveBench ≥1 hint (of 21) | domain ≥1 hint (of 120) | domain mean hints per prompt |
|---|---|---|---|
| 0.40 | 5 | 120 | 2.99 |
| 0.45 | 2 | 112 | 2.63 |
| **0.50** | **0** | **94** | **1.94** |
| **0.55 (served)** | **0** | **59** | **1.12** |
| 0.60 | 0 | 32 | 0.56 |

Per domain at 0.50 vs 0.55: typescript 16 vs 12 of 30, typegpu 25 vs 19 of
25, rust_wasm 21 vs 6 of 25, react 32 vs 22 of 40.

**Injection rate is not quality.** The closest thing to a quality proxy
available offline is selection precision on the 89 labelled hint probes
(`probe_floor_sweep.py`: the production path with the same cache and k = 3).
"Right member shown" is a precision count, not an outcome:

| floor | any hint shown | correct member shown | wrong sibling shown | nothing from the probe's bucket | total hints shown |
|---|---|---|---|---|---|
| 0.45 | 85 | 49 | 11 | 29 | 245 |
| **0.50** | 77 | **44** | **10** | 35 | 196 |
| **0.55 (served)** | 61 | **27** | **9** | 53 | 116 |
| 0.60 | 26 | 15 | 5 | 69 | 36 |

Going from 0.55 to 0.50 adds 17 correct showings for 1 more wrong sibling.
The share of shown bucket members that are correct moves from 0.75 to 0.82.
What this table cannot say is whether the roughly 80 extra non-bucket hints
at 0.50 are relevant, because those rows have no labels.

**The served number has drifted.** `hints.py` records 30 correct showings at
0.55 on a 938-row corpus. On today's 1,860 rows it is 27. More rows now
compete for the three slots.

### Why LiveBench cannot fire at any sensible floor [M]

- **What the nearest rows are.** For 19 of 21 prompts the nearest row comes
  from `complexity.jsonl`: 115 technique notes such as "n<=20: O(2^n) subset
  enumeration fits a one-second limit". LCB prompts are story-form problem
  statements, and a technique note is not similar to a story about a
  problem.
- **What 0.40 would inject.** Five prompts clear it, and I read each pair.
  "find the maximum number of elements in a subset" (n up to 10^5) gets the
  `n<=20` subset-enumeration note. "longest subarray with at most k
  frequency" gets a monotonic-deque minimum note. "count prefix and suffix
  pairs I" (tiny n) gets polynomial hashing. "maximize consecutive elements"
  gets LIS tails. Arguably one of the five fits. This is one reader's
  judgement, n=5.
- **Two diagnostics, neither of which changes the answer:**

  | variant | LiveBench ≥1 hint at 0.50 | effect on the domain suite |
  |---|---|---|
  | a recipe-shaped query instruction (`recipe_q`) | 1/21 (was 0) | maximum cosine rises from median 0.549 to 0.610; 89/120 injected at 0.55 |
  | only the problem text, with LiveBench's wrapper sections cut (`lb_core`) | 1/21 | – |

  The `recipe_q` gain is a uniform upward shift. It says nothing about
  precision.

  The production instruction was written for code search ("retrieve the
  source code that answers it"), not for recipes. That mismatch is real, but
  it is not why LiveBench is silent.

### Why the domain suite shows so few hints [M]

`domain_hint_records.py` pairs every domain-suite row that allowed hints with
its offline prediction:

- There are 12 such rows: A5 in `smoke-medium`, A6 overnight, and S6. They
  cover 6 tasks: ts01, ts02, tg01, tg02, rs01, rs03.
- On ts01, rs01 and rs03 the best cosines are 0.476, 0.506 and 0.500. The
  offline trace predicts 0 hints, and every one of those 9 production rows
  injected 0.
- On ts02, tg01 and tg02, production injected 2, 1 and 1 in `smoke-medium`,
  where the offline trace predicts 3 each. That run is older and used a
  smaller corpus.

The domain suite is not a low-firing substrate: 59/120 at 0.55. It has
simply not been run with hints on. That clears the stage-0 gate in
`DECISION-TREES-AND-SKILLS.md` §2.8, "stop if fewer than about 20% of tasks
get anything". LiveBench fails the same gate, at 0%.

### What to do [P]

1. Run A2 (hints only) against A0 on the 120 uncontaminated domain tasks, as
   stage 1 of that plan. Add a second hints arm at
   `YAMADORI_HINT_FLOOR=0.50`, which is an environment variable, not a code
   change. At 49–78% firing, the arms can produce discordant pairs.
   Pre-register two comparisons: A2@0.55 vs A0, and A2@0.50 vs A0.
2. Do not lower the served floor on the probe table alone. That table
   measures selection, and PROTOCOL rules 9–10 apply.
3. For LiveBench-style puzzles, the floor is not the lever. The algorithm
   rows would need keys phrased the way problems are stated ("the answer is
   a count over all subsets, n ≤ 20"), or a problem-to-technique step before
   retrieval. Until then, hints on those prompts would mostly be off-target.

---

## Q2. Code quality, bonsai vs yamadori-xhigh, the 21 LiveBench answers

### Method [M]

`code_quality.py` rebuilds every program as LiveBench's grader does
(`LCB_generation_process_results`): `extract_code` takes the last fenced
block, and for a completion question the starter is prepended unless the
extraction already begins with it. Per program it records:

- **ruff 0.16.8**, with a wide rule set: E, W, F, C90, N, B, SIM, C4, UP,
  PERF, RUF, PL, RET, ARG;
- **McCabe per function**, from ruff's C901 at `max-complexity 0` (radon is
  not installed);
- **a whole-program decision-point count**, from the AST, which also covers
  stdin scripts written at module level;
- **length**, in lines and characters of the model-written part;
- **runtime** on every public and private test, in a child process with a
  real stdin and stdout, timed around the call only. Each figure is the
  minimum of 5 runs for public tests and 3 for private.

`refine_q2.py` then drops findings that are not the model's doing:

- whitespace (W291–W293) and E501;
- `N802` and `UP004`, which LiveBench's starter code imposes;
- `F821` for names the grader supplies (it prepends `from typing import *`
  and similar lines).

It also checks compilation with `compile()` rather than `ast.parse`, because
`return` outside a function passes the parser and fails the compiler.

### Results, paired per question (xhigh minus bonsai) [M, n=21, one run]

| measure | bonsai | xhigh | xhigh higher / lower / same | sign p |
|---|---|---|---|---|
| does not compile as graded | 3 (5c9d7ade, 7878539e, 9e895f81; all completion) | 0 | – | – |
| lint findings the model is responsible for, total | 56 | 67 | 7 / 7 / 7 | 1.000 |
| idiom families (SIM, C4, UP, PERF, RUF, PL, B, RET) | 7 | 12 | 3 / 2 / 16 | 1.000 |
| correctness-class findings (F, B, syntax) | 7 | 6 | 3 / 5 / 13 | 0.727 |
| AST decision points, total | 130 | 145 | 7 / 5 / 8 | 0.774 |
| highest McCabe, summed over questions | 108 | 100 | 4 / 6 / 9 | 0.754 |
| characters, all 21 | 8,366 | 10,895 | 14 / 5 / 2 | 0.064 |
| characters, **completion** (n=10) | 1,526 | 4,468 | 8 / 0 / 2 | **0.008** |
| characters, generation (n=11) | 6,840 | 6,427 | 6 / 5 / 0 | 1.000 |

The findings differ in kind:

- **bonsai:** 2 × F706 (`return` outside a function) and one invalid-syntax
  error. These are its three non-compiling completions.
- **xhigh:** 3 × F811 (a redefinition, which is how a re-emitted class shows
  up), 8 × E702 (several statements joined with semicolons on one line), and
  2 × F841.
- **both:** a naming finding (N806) dominates in each arm.

**The mechanism behind the completion difference.** 6 of xhigh's 10
completion answers re-emit the whole `class Solution` rather than continuing
the starter. Bonsai does this on 0 of 10. The 6 are exactly the 6 where
fan-out delivered the tie-breaker's code (`replaced` true). The grader
appends the whole class after the starter, and Python lets the second
definition win, so a re-emitted class dodges the indentation and open
docstring traps that broke bonsai's continuations.

**Runtime** [M]. The 16 questions where both arms pass every private test:

- xhigh is more than 10% faster on 6, more than 10% slower on 1, and within
  10% on 9. Sign p = 0.125; geometric mean ratio 0.68.
- The mean is driven by two questions: e25750fd (1.4 ms vs 43.5 ms; xhigh's
  re-emitted class uses a different algorithm) and df00a468 (186 ms vs
  1,040 ms; xhigh's answer writes `sys.stdout.buffer` and is officially
  scored 0).
- Public tests are too small to separate anything (microseconds each).
- These times come from one measurement session on a machine also running
  the domain benchmark.

**n is too small** for everything in this section except the completion
length difference, and that one is explained by the re-emission rather than
by any quality of the code.

### The domain suite's style score (`bench/domain/grade_style.py`) [M]

**What exists.** ESLint `strict-type-checked` plus react-hooks plus React 19
idiom rules, for TypeScript and React answers only. Rust is skipped by
design (`kind 'rust' is not TypeScript`).

Across every results directory:

- **Lint scores:** 9 rows have one. All are ts01–ts04, and all come from arms
  A0 or S0.
- **t1a:** 1 of 2 rows is linted (A0 ts03, 0 errors).
- **t1b:** 0 of its 12 rows are linted. 11 are Rust and skipped, and 1 is
  an S0 Rust stack error with no style record.

**No stack-on row has a style score**, so the suite's style score cannot
answer Q2 today.

**What could be scored offline.** `style_smoke_pairs.py` applies
`grade_style.style()` to the only stack-on vs stack-off pairs in a covered
language: `smoke-medium` A0 vs A5 on ts01, ts02, tg01 and tg02.

| task | A0 lint errors (outcome) | A5 lint errors (outcome) | A5 hints injected |
|---|---|---|---|
| ts01 | 1 (pass) | 1 (pass) | none |
| ts02 | 1, a parse error (fail) | 1 (fail) | 2 |
| tg01 | 7, `no-unsafe-*` (fail) | 1 (fail) | 1 |
| tg02 | 8, `no-unsafe-*` (fail) | 1 (**pass**) | 1 |

A0's typegpu `no-unsafe-*` errors mean values the type checker could not
type. That fits calls to typegpu API that does not exist.

This is n=4, from a stack since marked stale, one run each. It is a
direction, not a result. To get a result, S5, A5 or A6 rows need to land on
TypeScript tasks. Adding a Rust linter (`cargo clippy`) would cover the rows
the suite actually has.

---

## Q3. Where the stack does worse, and a generation vs completion router

### The six discordant questions [M, n=21]

| id | task | bonsai | xhigh | what happened in xhigh |
|---|---|---|---|---|
| 388e39ee | LCB generation (stdin) | 1 | 0 | reads `sys.stdin.buffer`; passes 15/15 on a real stdin; fan-out 2 steps, `code_grade`, the **original** kept; no repair |
| cf932b62 | LCB generation (stdin) | 1 | 0 | same: `.buffer`, passes on a real stdin, original kept, 3 tool turns |
| df00a468 | LCB generation (stdin) | 1 | 0 | writes `sys.stdout.buffer`; passes on a real stdout; original kept |
| 45d51d09 | completion | 0 | 1 | the tie-breaker's code delivered (`code_medoid`, the only candidate that parses); it re-emits the class; repair 1 round, stopped at `repeated_answer`. Bonsai's answer compiled and failed the tests |
| 9e895f81 | completion | 0 | 1 | the tie-breaker's code delivered and re-emits the class. Bonsai: `return` outside a function |
| 7878539e | completion | 0 | 1 | the original kept (`fallback`: no candidate parses); **repair** 1 round, clean. Bonsai: `return` outside a function |

- **The three generation losses.** The stack's mechanisms did not cause
  them: the model's own original was kept each time, and the grader quirk
  zeroed it. Nothing in the record says why the main model writes `.buffer`
  in 4 of 11 LCB answers under xhigh, against 1 of 11 under bonsai. The
  prompts differ only by tier and by `check_code` being offered. That is a
  hypothesis, not a finding.
- **Fan-out on completion runs blind.** No completion candidate parses on
  its own, because each is a fragment. On 4 of the 10 questions none
  parsed, and on the other 6 only the tie-breaker's full class did. So
  selection was `code_medoid` or `fallback` on 10/10. On generation, every
  candidate parses on 10/11. The prefix-aware check that another agent is
  adding to `fanout.py` addresses exactly this.

### (a) The detector [M]

`split_detector.py` defines four signals. None calls a model.

- **`P_raw`:** the last fenced block fails to parse on its own, using
  `code_check.syntax_errors`, read-only: Python's compiler for Python,
  tree-sitter otherwise.
- **`P_stub`:** the block still fails after an indented `pass` is appended.
  A bare signature `def f(self, x):` is a generation scaffold, while an open
  docstring or an open expression stays broken.
- **`I_op`:** the operator's words (continue, complete, finish or fix, any
  inflection) anywhere outside code.
- **`I_narrow`:** phrases that treat the code as unfinished or broken:
  "remaining lines", "missing portion", "partial completion", "first lines
  of", "complete/finish/continue the \<code noun\>", "fix the \<code
  noun\>".

The rules combine them:

- **`R_op`** = `P_raw` or `I_op`: the operator's rule as stated.
- **`R_ref`** = `P_stub` or `I_narrow`.
- **`R_text`** = `I_narrow` only.

Ground truth:

- **LiveBench:** the task label, over all 128 questions of release
  2024-11-25.
- **Domain core and react:** by construction, each asks for a whole module
  from a spec, so each is negative.
- **three_tsl:** the 15 prompts that paste a complete old module and say
  "Port it" are edits, so positive. I labelled these from reading the
  prompts myself, as the only reader (PROTOCOL rule 7).

| rule | LiveBench, the 21 run | LiveBench, the 107 not run | domain uncontaminated, 120 (all negative) | three_tsl edits (15 of 20, contaminated) |
|---|---|---|---|---|
| **R_op (as stated)** | 16/21 (5 false positives) | 69/107 (38 false positives) | 109/120 (11 false positives) | 0/15 caught |
| R_ref | 21/21 | 107/107 | 120/120 | 0/15 caught |
| **R_text** | **21/21** | **107/107** | **120/120** | 0/15 caught |
| `P_raw` alone | 9/21 | 48/107 | 120/120 | 0/15 |
| `P_stub` alone | 12/21 (1 of 10 completions caught) | 74/107 (7 of 40) | 120/120 | 0/15 |

**Why the operator's rule misfires.**

- 40 of LiveBench's 78 generation prompts give a LeetCode stub such as
  `class Solution:` / `def f(self, x) -> int:` with no body. The stub fails
  to parse, so `P_raw` flags it.
- Most completion starters are cut at a statement boundary, so they parse.
  An example is a signature plus docstring plus one statement. `P_raw`
  catches only 2 of the 10 completions in the 21.
- The words "complete" and "fix" also appear in problem statements ("fixed
  uniform binding", "complete binary tree").

**The parse clause adds nothing measurable.** `P_stub` caught 1 and 7
completions, and `I_narrow` caught every one of them anyway.

**Limits of the 128/128.**

- LiveBench uses exactly two instruction templates, one for 78 prompts and
  one for 50. So the 128/128 is **2 distinct cases**, not evidence that the
  rule generalises.
- Ports (three_tsl) are missed by every rule as specified, because they
  paste complete, parseable code and say "port". Adding "no longer builds",
  "port it" and "migrate" catches 14/15 with 1 false positive in 120. I
  chose those words after reading these prompts, so that result is not
  validated.
- The 183 type-challenge prompts ("Start from this template ... containing
  the complete implementation") are flagged by `R_text` because the phrase
  matches "complete implementation". Requiring an article after the verb
  ("complete **the** code") clears all 183 and keeps LiveBench at 128/128.
  That is also a post-hoc change.
- None of these prompts is real client traffic. A Hermes turn that says "fix
  the failing test" carries files and diffs, not LiveBench's template.
  Before this rule gates anything, it has to be scored on prompts captured
  by `corpus.log_turn`, labelled by someone who did not write the rule
  (PROTOCOL rule 7).

### (b) Arm results by class [M, n=21, one run]

The detector `R_text` agrees with the true label on all 21, so the two
breakdowns are identical.

| class | scoring | bonsai | xhigh | xhigh only / bonsai only | exact McNemar p |
|---|---|---|---|---|---|
| generation (n=11) | official | 9 | 6 | 0 / 3 | 0.250 |
| generation (n=11) | grader given `.buffer` | 10 | 10 | 0 / 0 | 1.000 |
| completion (n=10) | official | 6 | 9 | 3 / 0 | 0.250 |
| completion (n=10) | grader given `.buffer` | 6 | 9 | 3 / 0 | 0.250 |

What xhigh costs [M]:

| class | median output tokens, bonsai vs xhigh | ratio |
|---|---|---|
| generation | 1,545 vs 10,540 | 6.8× |
| completion | 3,876 vs 10,037 | 2.6× |

Wall-clock times fall in different timing epochs (`condition.json`), so they
are not compared.

**What the data supports, at this n.**

- **Generation:** the stack changed no outcome that is not a grader quirk,
  and it cost about 7× the tokens. That makes it a cost-based candidate for
  "stack off". It is not an accuracy finding.
- **Completion:** the direction favours the stack, 3 vs 0. The mechanism is
  format repair, meaning whole-class re-emission and one repair round, not
  better algorithms. Once the fan-out check reads the starter prefix, this
  gain may shrink or grow, so re-measure after that change lands.

### (c) The n needed to decide "stack on for class X" [M, arithmetic; `power.py`]

**Assumptions.** Exact two-sided McNemar, one sample per arm per question. A
question is discordant with probability d, and a discordant question favours
the stack with probability π. The observed discordance is 3/10 and 3/11 per
class, and the bare-vs-bare noise is 8/21 = 0.38, so d = 0.30–0.38 is the
realistic range. The 3-0 splits give no usable estimate of π.

**Minimum discordant pairs.** Even with every discordant pair going one way,
significance needs 6 pairs at α = 0.05 and 7 at α = 0.025. Both classes today
are at 3-0, p = 0.25.

**n per class for 80% power:**

| d | π | net gain (d·(2π−1)) | at α = 0.05 | at α = 0.025 (two classes, Bonferroni) |
|---|---|---|---|---|
| 0.30 | 0.75 | 15 points | 112 | 131 |
| 0.30 | 0.85 | 21 points | 55 | 65 |
| 0.38 | 0.75 | 19 points | 88 | 103 |
| 0.38 | 0.85 | 27 points | 43 | 51 |
| 0.38 | 0.95 | 34 points | 25 | 28 |

**Power the clean held-out pool gives (below), at α = 0.025.**

| class | clean n | power, π = 0.75 | power, π = 0.85 |
|---|---|---|---|
| completion | 37 | 0.20–0.28 | 0.47–0.62 |
| generation | 60 | 0.38–0.50 | 0.76–0.88 |

Each range spans d = 0.30 to 0.38. **Say in advance that a null on
completion is the likely outcome, unless the true effect is large.**

### (d) The held-out validation set [M counts, P design]

**What LiveBench has.** Release 2024-11-25 has 128 coding questions in these
two tasks: 78 `LCB_generation` and 50 `coding_completion`. We ran 21 of them
(11 and 10), which leaves **67 generation and 40 completion unused**. Later
releases publish none (`count_holdout.py`: 0 for every release from
2025-04-02 on).

**Leakage.** Completion questions are built from LCB problems. 32 problems
appear in both tasks, and 10 of the unused questions share a problem with the
21 we ran. Dropping those leaves **60 generation and 37 completion clean**
(31 stdin and 29 functional on the generation side; 66 LeetCode and 31
AtCoder).

**Tuning on the 21 and reporting on them is overfitting.** The detector's
wording, the class split and the "stack off for generation" idea were all
formed after looking at these 21 results. From now on they are development
data. No threshold, phrase or policy may be adjusted on them and then
reported on them.

**Proposed design [P]:**

1. **Freeze first.** Freeze `R_text` as written (keep the article-free form,
   or pre-register the article form now) and the policy "stack on for
   completion, off for generation". Write both into the run directory before
   any held-out answer exists.
2. **Primary comparisons.** Run both arms, bonsai and xhigh, on the 97 clean
   questions, one request at a time through `bench/queue_runner.py`. There
   are two pre-registered comparisons, xhigh vs bonsai within each class,
   exact McNemar at α = 0.025 each. Report `.buffer` sensitivity beside the
   official score, never instead of it.
3. **Policy row.** Also score the routed policy against always-on and
   always-off, as a secondary row only.
4. **Buying power.** The completion class cannot reach significance on 37
   items unless the effect is huge. Buy power instead of pretending:
   - **Repeat samples.** Draw 3 samples per arm per question, which is legal
     since the sampling is fixed, and test per question with the pass count
     paired (sign or Wilcoxon on the per-question differences). This lowers
     the noise that the 8/21 self-disagreement shows.
   - **More completion items.** `bench/data/test5.jsonl` and `test6.jsonl`
     hold 342 more LCB problems, with private tests, from September 2024 to
     April 2025. They have no reference solutions. So completion items would
     have to be cut, the way LiveBench cut them, from solutions from a
     source other than either arm, to avoid favouring one arm's style. They
     would also need their own leak check against these problem titles.
5. **Label the domain as contaminated.** Every LiveBench and LCB problem
   predates the Qwen3.8 base, so contamination is possible across the whole
   domain.

**Cut criterion [P].** Route "stack off for generation" only if one of these
holds:

- the generation comparison shows no stack win at α = 0.025, and the token
  cost stays at least 3×; or
- the stack loses on generation with `.buffer` provided.

Route "stack on for completion" only if xhigh beats bonsai there at
α = 0.025. Otherwise, leave today's per-tier behaviour alone and record the
null.

---

## Discrepancies found (reported, not edited)

| where | what it says | what is true |
|---|---|---|
| `mcp/hints.py` docstring | 938 recipes; correct member shown 30/89 at 0.55 | 1,860 served rows (179 rejects excluded); 27/89 today (`probe_floor_sweep.py`) |
| `selection.decide` reason | "attaches only what clears the 0.55 floor inside the task's domains" | the domain filter never engaged on any of 141 benchmark prompts: none carries strong evidence |
| `hints.attach` → `task_domains(messages)` | reads the whole conversation | when the code tools are offered, `messages` includes our capability block. Its words add `gpu`, `brand` and `backend` to `domains.detect()` whenever a prompt also carries strong evidence. That only ever widens the filter, and it is latent today (no prompt has strong evidence) |
| `hints.select` query | `code_search.QUERY_INSTRUCT`, "retrieve the source code that answers it" | the instruction is for code search. A recipe-shaped instruction shifts every cosine up by about 0.06 on the domain suite. Its precision is unmeasured |
| `lb-20260923-minp0/sanity.md` | bonsai's 3 completion zeros are SyntaxErrors | confirmed with `compile()`: 5c9d7ade (open docstring), 7878539e and 9e895f81 (`return` outside a function). `ast.parse` accepts the last two |

## Files

Under `bench/analysis/stack_tuning_0923/`, all run with the stack
interpreter unless noted. Nothing here is imported by the stack.

| file | what it does |
|---|---|
| `dump_livebench.py` | WSL venv: dumps the release questions and both arms' answers into `data/` |
| `count_holdout.py` | WSL venv: coding questions per release |
| `hint_trace.py` | Q1 production-path trace (309 prompt embeddings, parity with `hints.select`) |
| `hint_report.py` | stage and floor tables |
| `domain_hint_records.py` | domain rows against the prediction |
| `probe_floor_sweep.py` | Q1 selection precision per floor (89 embeddings) |
| `code_quality.py` | Q2 lint, complexity, length, runtime |
| `refine_q2.py` | model-attributable lint, compile check, re-emission |
| `style_smoke_pairs.py` | `grade_style` on the four A0/A5 TS pairs |
| `split_detector.py` | Q3 detector and accuracy |
| `summarise_q2_q3.py` | paired and per-class tables |
| `power.py` | McNemar power |

`data/` holds the outputs. It also holds the LiveBench question text and both
arms' answers, so check the licence before committing it. The full question
dump with private tests (255 MB) is in the session scratchpad, not the repo.
