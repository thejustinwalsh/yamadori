# Sanity checks, run lb-20260923 (bonsai arm)

Per docs/PROTOCOL.md rule 3: a zero is plumbing until shown otherwise.

## Harness and scorer (before any model answer was trusted)

- Header path: every answer carries `api_info.x_yamadori` with the arm's flags
  in `selection.signals.forced`; `drive.py` stops the run on a mismatch. No
  mismatch occurred.
- Scored field: `choices[0].turns` holds `content`; the thinking is kept
  separately under `choices[0].reasoning` (upstream harness behaviour).
- Judge self-test (`validate_scorer.py`): 33/33. Ground-truth-formatted right
  answers score 1 and wrong answers score below 1 for every non-coding task;
  IF scores land in [0, 1] with no eval_error. It found a crash in LiveBench's
  old-format IF path (patched, see README).
- Reproduction: the published 2024-11-25 table is recomputed from LiveBench's
  per-question judgments (`comparison.md`, reproduction table). It matches to
  0.0 points per task for 14 of 18 reference models. The exceptions are
  o3-mini-high 1.3, DeepSeek R1 3.2, QwQ-32B 4.3 (connections) and
  R1-Distill-Qwen-32B 17.4 (paraphrase). These are re-judged rows in the HF
  set, which affects only the reference rows, never ours.

## Coding, first 11 answers (read at 03:30)

- finish_reason: 11/11 `stop`; no `length`. Output tokens 121 to 13,355;
  the thinking budget sent was ~97k, so no answer came near it.
- LCB_generation 6/6. coding_completion 2/5. All three zeros were rebuilt
  exactly as the judge runs them (`partial_solution + "\n" + answer`):
  - 3cd8c16a (earliest-second-to-mark-indices): the completion is indented 8
    spaces where the `while` body needs 12, so `if check(mid)` falls outside
    the loop and the program never terminates (judge timeout). An
    independently written canonical completion passes all 15 tests in 0.0 s
    on the same grader, so the grading machine is not the cause. 100/166
    published models pass this question.
  - 45d51d09 (maximum-number-of-elements-in-subset): the completion is
    indented 4 spaces inside a block 12 deep, a SyntaxError
    (`return` outside function). 39/166 published models pass.
  - 5c9d7ade (ant-on-the-boundary): the starter code ends inside an opened
    docstring (`"""`) and the completion never closes it, a SyntaxError.
    16/166 published models pass.
  The prompt says: "Only write the missing portion ... Be very careful to
  match the appropriate indentation." These are model errors under the
  benchmark's rules, not plumbing: 2 of 3 are indentation of a continuation.
- Transport: one 429 ("all 2 main lanes busy") before the proxy moved to 4
  lanes. It was logged in `errors.jsonl`, retried and answered. No question
  is scored from an error row.

## Coding, all 21 answers of the stage (read at 05:50)

- finish_reason: 21/21 `stop`; no `length`; no error rows in the scored set.
- 6 zeros, all rebuilt and diagnosed as model errors:
  - coding_completion: the three above, plus
    - eae49210 (minimum-number-of-coins-to-be-added): the completion is
      indented 4 and 0 spaces under a `while` that needs 12, a SyntaxError
      ("expected an indented block"). 115/166 published models pass.
    - e25750fd (count-beautiful-substrings-i): compiles, fails the tests.
      32/166 pass.
  - LCB_generation df00a468 (atcoder Many Replacement): the algorithm is
    right, but the output line is `chr(mapping[...])` without `+ ord('a')`,
    so it prints control characters. All 15 tests are wrong in 0.02-0.12 s,
    so this is not a timeout on our grading machine. 18/166 pass.
- Pattern: 4 of the 5 coding_completion zeros are continuation-format
  failures (indentation of the appended code, or an unclosed docstring),
  not wrong algorithms. That is what the task measures, so they stand.

## Instruction following, first 4 answers (read at 06:20)

- finish_reason 4/4 `stop`. Answers are plain `content` (no thinking text
  leaked in): 163-357 words, 452-1,225 output tokens.
- The scoring rule was read in `process_results/instruction_following/utils.py`:
  question score = (all instructions followed ? 1 : 0 + fraction followed) / 2,
  strict IFEval checkers. This matches LiveBench's published IF numbers.
- paraphrase (wrap in double quotes), simplify (forbidden words, P.P.S,
  under 15 sentences, exactly 2 bullets) and summarize (P.P.S, 4 paragraphs
  split by `***`) score 1.0, and each was checked by eye against its
  instructions.
- story_generation 1b09cc81 scores 1/3: JSON wrapper and required keywords
  pass. The answer then adds a `"keywords_excluded"` field listing the
  forbidden words verbatim, which fails `forbidden_words`, so
  (0 + 2/3) / 2 = 1/3. A genuine model error, and the judge is right.

## Throughput caveat

The GPU was shared with other benchmarks all night: 4 of 4 slots busy,
~6.4 tok/s for our request against ~25 tok/s uncontended. `seconds` in the
rows is wall time under that contention and is not a latency measurement.
