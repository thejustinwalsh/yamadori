# yamadori vs bonsai (bare @ medium), coding, run lb-20260923-minp0

Arms: `bonsai` = bare @ medium (all augmentation forced off, effort medium); `yamadori` = tier max (reasoning_effort max -> xhigh upstream; selection decides fan-out/deep thinking). The tier also raises the thinking effort sent upstream, so this pair mixes effort and tools; it is a product comparison, not a tool ablation.

## Official scores (LiveBench judge, unmodified)

| arm | score [95% CI] | n | per task |
|---|---|---|---|
| bonsai | 70.9 [51.4, 90.0] | 21 | LCB_generation 81.8 (n=11), coding_completion 60.0 (n=10) |
| yamadori | 100.0 [100.0, 100.0] | 1 | LCB_generation 100.0 (n=1) |

**Paired (yamadori minus bonsai), n=1 questions both arms scored:** +0.0 points [95% CI +0.0, +0.0]. Discordant pairs: yamadori right only 0, bonsai right only 0; exact McNemar p = --.

**stdin.buffer sensitivity** (LiveBench's LCB grader gives programs a StringIO stdin without `.buffer`; same answers re-graded with `.buffer` provided, `scan_stdin_buffer.py`): bonsai 75.5, yamadori 100.0; answers affected: 1. Official scores above stand.

## Per question

| # | id | task | bonsai | yamadori | fan-out (mode, steps, stop, method, winner, replaced) | deep thinking (decided/ran) | repair | status | yamadori timing |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 27c31c63 | LCB_generation | 1 | 1 | sequential, 3, tie-breaker: both parse but disagree, code_medoid, original, False | False/False | 0 rounds, clean | ok/ok | single_request_enforced |

Deep thinking (investigate) decided on 0 of 1 questions and ran on 0. Selection's reason when it did not is in `x_yamadori.selection.because.investigate` on each answer.
