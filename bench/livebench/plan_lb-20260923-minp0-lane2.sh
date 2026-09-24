#!/bin/bash
# Lane 2 of lb-20260923-minp0 after GO (09:03 EDT): the code-task fan-out at
# tiers high/max is deployed (proxy restart 09:02:36), so tier max runs first,
# as the main augmented arm. It starts only after the n=1 re-gate passed
# (decided_n=3, selection code_medoid, winner delivery and candidate parses
# recorded). Then the arms that do not depend on the change.
#   tier max coding (21) -> bonsai+check coding (21, resumes its 1 answer) -> tier minimal coding (21)
# History: ~08:52 this lane ran bonsai+check first while tier max waited for
# the deploy; it was reordered at GO.
#   KEY_FILE=/path/to/key bash plan_lb-20260923-minp0-lane2.sh
set -u
HERE=/mnt/c/Users/jwals/llama-stack/bench/livebench
RUN=lb-20260923-minp0
for arm in yamadori bonsai+check minimal; do
  echo "=== $(date +%T) lane 2: $arm coding n=21"
  bash "$HERE/run_arm.sh" "$arm" coding $RUN 21 0
done
echo "=== $(date +%T) lane 2: done"
