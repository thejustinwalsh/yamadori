# yamadori-xhigh vs bonsai (bare @ medium), coding, run lb-20260923-minp0

Arms: `bonsai` = bare @ medium (all augmentation forced off, effort medium); `yamadori-xhigh` = tier xhigh (reasoning_effort xhigh -> MEDIUM upstream; every augmentation allowed: retrieval, hints, fan-out 3, deep thinking, check_code, repair, 10-turn tool cap) -- the all-on arm, effort-matched to bonsai. Both arms send MEDIUM thinking effort upstream, so this pair is effort-matched: all augmentation allowed vs all forced off.

## Official scores (LiveBench judge, unmodified)

| arm | score [95% CI] | n | per task |
|---|---|---|---|
| bonsai | 70.9 [51.4, 90.0] | 21 | LCB_generation 81.8 (n=11), coding_completion 60.0 (n=10) |
| yamadori-xhigh | 72.3 [54.1, 90.5] | 21 | LCB_generation 54.5 (n=11), coding_completion 90.0 (n=10) |

**Paired (yamadori-xhigh minus bonsai), n=21 questions both arms scored:** +1.4 points [95% CI -17.7, +20.5]. Discordant pairs: yamadori-xhigh right only 3, bonsai right only 3; exact McNemar p = 1.000.

**stdin.buffer sensitivity** (LiveBench's LCB grader gives programs a StringIO stdin without `.buffer`; same answers re-graded with `.buffer` provided, `scan_stdin_buffer.py`): bonsai 75.5, yamadori-xhigh 90.5; answers affected: 5. Official scores above stand.

## Per question

| # | id | task | bonsai | yamadori-xhigh | seconds (bonsai / yamadori-xhigh) | fan-out (mode, steps, stop, method, winner, replaced) | deep thinking (decided/ran) | repair | tool turns (turns/limit, hit) | status | yamadori-xhigh timing |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 27c31c63 | LCB_generation | 1 | 1 | 6.3 / 16.0 | sequential, 3, tie-breaker: both parse but disagree, code_medoid, tiebreak, True | False/False | 0 rounds, clean | 0/10, False | ok/ok | single_request_enforced |
| 2 | ebde77f9 | coding_completion | 1 | 1 | 453.8 / 306.5 | sequential, 3, tie-breaker: neither parses, fallback, original, False | False/False | 0 rounds, clean | 1/10, False | ok/ok | single_request_enforced |
| 3 | 3375d42f | LCB_generation | 0 | 0 | 455.9 / 1766.2 | sequential, 3, tie-breaker: both parse but disagree, code_medoid, tiebreak, True | False/False | 0 rounds, clean | 0/10, False | ok/ok | single_request_enforced |
| 4 | 3cd8c16a | coding_completion | 1 | 1 | 402.3 / 279.3 | sequential, 3, tie-breaker: neither parses, code_medoid, tiebreak, True | False/False | 0 rounds, clean | 0/10, False | ok/ok | single_request_enforced |
| 5 | 6afe4974 | LCB_generation | 1 | 1 | 21.4 / 18.2 | sequential, 3, tie-breaker: both parse but disagree, code_medoid, tiebreak, True | False/False | 0 rounds, clean | 0/10, False | ok/ok | single_request_enforced |
| 6 | 45d51d09 | coding_completion | 0 | 1 | 461.4 / 514.7 | sequential, 3, tie-breaker: neither parses, code_medoid, tiebreak, True | False/False | 1 rounds, repeated_answer | 0/10, False | ok/ok | single_request_enforced |
| 7 | 89101e97 | LCB_generation | 1 | 1 | 49.3 / 49.2 | sequential, 3, tie-breaker: both parse but disagree, code_medoid, tiebreak, True | False/False | 0 rounds, clean | 0/10, False | ok/ok | single_request_enforced |
| 8 | 7878539e | coding_completion | 0 | 1 | 260.6 / 308.4 | sequential, 3, tie-breaker: neither parses, fallback, original, False | False/False | 1 rounds, clean | 0/10, False | ok/ok | single_request_enforced |
| 9 | d5213885 | LCB_generation | 1 | 1 | 540.8 / 315.5 | sequential, 3, tie-breaker: both parse but disagree, code_medoid, tiebreak, True | False/False | 0 rounds, clean | 0/10, False | ok/ok | single_request_enforced |
| 10 | 5c9d7ade | coding_completion | 0 | 0 | 31.9 / 26.8 | sequential, 3, tie-breaker: neither parses, code_medoid, tiebreak, True | False/False | 0 rounds, clean | 0/10, False | ok/ok | single_request_enforced |
| 11 | d414f776 | LCB_generation | 1 | 1 | 75.9 / 56.8 | sequential, 2, clear: agreement, code_grade, original, False | False/False | 0 rounds, clean | 0/10, False | ok/ok | single_request_enforced |
| 12 | eae49210 | coding_completion | 1 | 1 | 42.1 / 108.8 | sequential, 3, tie-breaker: neither parses, fallback, original, False | False/False | 1 rounds, clean | 0/10, False | ok/ok | single_request_enforced |
| 13 | e1e226d2 | LCB_generation | 0 | 0 | 389.6 / 441.1 | sequential, 3, tie-breaker: both parse but disagree, code_medoid, original, False | False/False | 0 rounds, clean | 0/10, False | ok/ok | single_request_enforced |
| 14 | e25750fd | coding_completion | 1 | 1 | 655.6 / 418.9 | sequential, 3, tie-breaker: neither parses, code_medoid, tiebreak, True | False/False | 0 rounds, clean | 0/10, False | ok/ok | single_request_enforced |
| 15 | df00a468 | LCB_generation | 1 | 0 | 237.2 / 263.3 | sequential, 2, clear: agreement, code_grade, original, False | False/False | 0 rounds, clean | 1/10, False | ok/ok | single_request_enforced |
| 16 | 786432dd | coding_completion | 1 | 1 | 10.7 / 240.9 | sequential, 3, tie-breaker: neither parses, fallback, original, False | False/False | 0 rounds, clean | 0/10, False | ok/ok | single_request_enforced |
| 17 | cf932b62 | LCB_generation | 1 | 0 | 290.4 / 471.8 | sequential, 2, clear: only one parses, code_grade, original, False | False/False | 0 rounds, clean | 3/10, False | ok/ok | single_request_enforced |
| 18 | 4952a5d4 | coding_completion | 1 | 1 | 49.4 / 79.2 | sequential, 3, tie-breaker: neither parses, code_medoid, tiebreak, True | False/False | 0 rounds, clean | 0/10, False | ok/ok | single_request_enforced |
| 19 | 388e39ee | LCB_generation | 1 | 0 | 184.2 / 383.7 | sequential, 2, clear: agreement, code_grade, original, False | False/False | 0 rounds, clean | 1/10, False | ok/ok | single_request_enforced |
| 20 | 9e895f81 | coding_completion | 0 | 1 | 119.4 / 398.4 | sequential, 3, tie-breaker: neither parses, code_medoid, tiebreak, True | False/False | 0 rounds, clean | 0/10, False | ok/ok | single_request_enforced |
| 21 | 87871f3e | LCB_generation | 1 | 1 | 20.4 / 21.1 | sequential, 2, clear: agreement, code_grade, original, False | False/False | 0 rounds, clean | 0/10, False | ok/ok | single_request_enforced |

Deep thinking (investigate) decided on 0 of 21 questions and ran on 0. Selection's reason when it did not is in `x_yamadori.selection.because.investigate` on each answer.
