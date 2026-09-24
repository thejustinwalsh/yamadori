# Sanity checks, run lb-20260923-minp0 (server min_p 0.0)

Per docs/PROTOCOL.md rule 3: a zero is plumbing until shown otherwise. Harness
and judge checks from `lb-20260923/sanity.md` still hold: the same patch, the
judge self-test at 33/33, and the reproduction of the published table.

## Bonsai (bare @ medium) coding, 21/21, completed 10:10 EDT

- finish_reason: 21/21 `stop`. No `length`, no stack errors in the scored set.
  Output tokens 138-12,050; the thinking budget sent was ~97k.
- One `incomplete` answer (cf932b62, the llama-swap restart at 09:57:39,
  "[the connection to the model dropped after 428s ...]") was caught, kept
  in `stack_errors.jsonl` and re-asked. It is not in the score. That is
  when `incomplete` became a stack error in drive.py and score.py.
- Official score 70.9 [51.4, 90.0]: LCB_generation 9/11 = 81.8,
  coding_completion 6/10 = 60.0.
- The 6 zeros, each rebuilt and run as the judge runs it (diag in the
  session log):
  - coding_completion 5c9d7ade, 7878539e, 9e895f81: SyntaxError when
    appended to the starter code. One is an unclosed starter docstring, two
    are `return` outside function from wrong indentation. The same
    continuation-format failure mode as the min_p 0.05 record.
  - coding_completion 45d51d09: compiles, fails the tests.
  - LCB 3375d42f (maximum-xor-product, functional): wrong greedy; fails the
    tests.
  - **LCB e1e226d2 (atcoder Loong Tracking): grader quirk.** LiveBench's LCB
    grader replaces sys.stdin with an io.StringIO that has no `.buffer`. The
    program reads `sys.stdin.buffer.readline` and dies with AttributeError
    (error_code -4) on test 1. On a real stdin it passes 13/13 tests, each
    under the 6 s limit. Given a `.buffer` holding the same bytes
    (`scan_stdin_buffer.py`), it scores 1. **The official 0 stands**, since
    every leaderboard model was graded by the same code. The sensitivity
    number is 75.5 (LCB 10/11), reported next to it and never instead of
    it. `stdin_buffer_sensitivity.json` has every such answer for every arm.

## Grader quirk: stdin/stdout `.buffer` (both arms, read 13:40)

LiveBench's LCB grader runs stdin programs with sys.stdin and the captured
sys.stdout replaced by io.StringIO objects that have no `.buffer`. A
correct program that reads `sys.stdin.buffer` or writes
`sys.stdout.buffer` dies with AttributeError (error_code -4) on test 1 and
scores 0. Every scored zero on an LCB stdin problem was re-run on a real
stdin/stdout. At 18 xhigh answers:

| arm | id | official | real stdin | `.buffer` provided (scan_stdin_buffer.py) |
|---|---|---|---|---|
| bonsai | e1e226d2 | 0 | 13/13 tests pass | 1 |
| yamadori-xhigh | e1e226d2 | 0 | passes | 1 |
| yamadori-xhigh | 388e39ee | 0 | 15/15 pass | 1 |
| yamadori-xhigh | cf932b62 | 0 | 15/15 pass | 1 |
| yamadori-xhigh | df00a468 (writes stdout.buffer) | 0 | 15/15 pass | 1 |

The official scores stand, because every leaderboard model was graded by
the same code. But the quirk falls unevenly on these arms: xhigh used
`.buffer` in 4 of its 10 LCB answers against bonsai's 1 of 11, and all 3 of
xhigh's losses to bonsai at 18 questions are this quirk. In each, the main
brain's original answer was kept (fan-out 2 steps), so the model wrote
`.buffer` itself. Nothing in the record says why it does so more often in
this arm. Evidence, not an explanation. The first scan matched only
`sys.stdin.buffer` and missed df00a468 (stdout). Fixed and re-run.

## yamadori-xhigh coding, 21/21, completed 13:44 EDT

- finish_reason 21/21 `stop`. Stack errors 0. tool_turns max 3/10, never hit.
- Official 72.3 [54.1, 90.5]: LCB_generation 6/11 = 54.5, coding_completion
  9/10 = 90.0.
- Its 6 zeros, each rebuilt and run:
  - e1e226d2, 388e39ee, cf932b62, df00a468: the `.buffer` grader quirk
    (above). All four pass on a real stdin/stdout.
  - 3375d42f (maximum-xor-product): wrong answer on a hidden test (322 vs
    330). The bonsai answer and the tier-max tie-breaker answer to this
    question differ.
  - 5c9d7ade (ant-on-the-boundary): the model rewrote the class and put the
    whole body inside the starter's open docstring, so the function returns
    None. Bonsai failed the same question another way (docstring left open).
- The final full-run numbers are in `summary.json`,
  `paired_yamadori-xhigh_vs_bonsai_coding.md` and `comparison.md`.

## Tier max: partial arm (dropped by the operator ~11:15)

- 2 answered rows after the 10:25:12 deploy (sequential fan-out, MAIN_LANES=1):
  the gate (LCB 27c31c63, 60 s, 503 tokens) and LCB 3375d42f.
- 3375d42f ran from 10:27:28 to 11:44:32 = 4,619 s, 68,675 output tokens,
  196,568 characters of thinking, finish `stop`. It was not killed. The
  coordinator had planned to record it as "runaway, killed at deploy", but
  it finished before any deploy, so it is recorded as what it is. Fan-out:
  sequential, 3 steps, tie-breaker, code_medoid, the tie-breaker's code
  delivered (`replaced` true). This is one answer's evidence for the
  concern that led the operator to move the all-on arm to MEDIUM effort
  (tier `xhigh`): at effort xhigh the main context thought for 60k+
  tokens. n=1; not a measurement.
- The chunk's other question (ebde77f9) was never sent. A 2 ms watcher
  killed the harness after the row landed.
- Earlier tier-max rows (pre-change 08:36, pre-sequential 09:03) are set
  aside in their folders and never reported.

## Run-to-run evidence (same 21 questions, bare arm)

min_p 0.05 record: 70.5 (15/21 right). min_p 0.0: 70.9 (15/21 right). The
two runs disagree on 8 of 21 questions: 4 right only at 0.05, 4 right only
at 0.0. Exact McNemar p = 1.0. The runs also differ in contention and time
of day. This is sampling variance at temperature 1.0, not an effect of
min_p, and it shows what one run of n=21 can resolve (PROTOCOL rule 10).
