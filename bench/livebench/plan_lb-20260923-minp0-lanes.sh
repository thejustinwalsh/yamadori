#!/bin/bash
# Run lb-20260923-minp0 after the 08:3x deploy (feature freeze from here on).
# Two lanes = 2 requests in flight in total (coordinator's budget):
#   lane 1: bonsai coding (the 21; already running when this was written) -> bonsai instruction_following (all 200)
#   lane 2: tier max coding (21) -> bonsai+check coding (21) -> tier minimal coding (21)
# Every arm runs to completion (no deadlines). Coding arms all use the same 21 questions (--n 21 of the fixed order).
#   KEY_FILE=/path/to/key bash plan_lb-20260923-minp0-lanes.sh 1|2
set -u
HERE=/mnt/c/Users/jwals/llama-stack/bench/livebench
RUN=lb-20260923-minp0
export LB_PARALLEL_FILE=$HERE/results/$RUN/parallel_bonsai_if.txt
case "$1" in
  1) while pgrep -f "[d]rive.py --arm bonsai --category coding" >/dev/null; do sleep 20; done
     echo "=== $(date +%T) lane 1: bonsai instruction_following"
     bash "$HERE/run_arm.sh" bonsai instruction_following $RUN 0 0
     echo "=== $(date +%T) lane 1: done" ;;
  2) unset LB_PARALLEL_FILE
     for arm in yamadori bonsai+check minimal; do
       echo "=== $(date +%T) lane 2: $arm coding n=21"
       bash "$HERE/run_arm.sh" "$arm" coding $RUN 21 0
     done
     echo "=== $(date +%T) lane 2: done" ;;
esac
