#!/bin/bash
# Run lb-20260923-minp0 (server min_p 0.0 from ~07:16 EDT, operator decision).
# Two streams, one request in flight each = 2 in flight (coordinator: SWE-bench
# stopped, share only with the domain suite):
#   A: bonsai   coding (the same 21 questions as the min_p 0.05 record) -> bonsai instruction_following from question 1
#   B: yamadori coding (the same 21), paired with A's coding
# Running the two coding arms at the same time also puts the pair under the
# same contention.
#   KEY_FILE=/path/to/key bash plan_lb-20260923-minp0.sh A|B
set -u
HERE=/mnt/c/Users/jwals/llama-stack/bench/livebench
RUN=lb-20260923-minp0
case "$1" in
  A) echo "=== $(date +%T) A: bonsai coding n=21"
     bash "$HERE/run_arm.sh" bonsai coding $RUN 21 0
     echo "=== $(date +%T) A: bonsai coding done; bonsai instruction_following"
     bash "$HERE/run_arm.sh" bonsai instruction_following $RUN 0 0
     echo "=== $(date +%T) A: done" ;;
  B) echo "=== $(date +%T) B: yamadori coding n=21"
     bash "$HERE/run_arm.sh" yamadori coding $RUN 21 0
     echo "=== $(date +%T) B: done" ;;
esac
