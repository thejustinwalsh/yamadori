#!/bin/bash
# lb-20260923-minp0 from ~09:43 EDT: ONE request in flight in total (operator: every
# benchmark request runs as it would for a single user, main + one second brain,
# full budgets; the proxy will enforce MAIN_LANES=1 from the next deploy).
#
# Order (coordinator, ~09:50: the paired coding comparison is the priority):
#   0. bonsai coding (the 21)       -- finished by this script (resumes from disk)
#   1. tier max coding              -- as soon as GO_TIER_MAX exists: gate n=1, wait for
#                                      GATE_OK (created by hand after inspection), then all 21
#   2. bonsai+check coding (21)     -- runs while tier max is not yet GO, so the lane stays on coding
#   3. tier minimal coding (21)
#   4. bonsai instruction_following (200)
# Every later arm runs in slices (2 questions = one chunk for coding, 8 for IF), and
# tier max is slotted in at the first slice boundary after GO_TIER_MAX appears.
#   KEY_FILE=/path/to/key bash plan_lb-20260923-minp0-single.sh
set -u
HERE=/mnt/c/Users/jwals/llama-stack/bench/livebench
RUN=lb-20260923-minp0
R=$HERE/results/$RUN
export LB_PARALLEL_FILE=$R/parallel_bonsai_if.txt   # holds 1
echo 1 > "$LB_PARALLEL_FILE"
TIER_MAX_DONE=0

maybe_tier_max() {
  [ "$TIER_MAX_DONE" = 1 ] && return
  [ -f "$R/GO_TIER_MAX" ] || return
  if [ ! -f "$R/GATE_OK" ]; then
    echo "=== $(date +%T) tier max gate (n=1)"
    bash "$HERE/run_arm.sh" yamadori coding $RUN 1 0
    echo "=== $(date +%T) tier max gate answered; waiting for GATE_OK"
    while [ ! -f "$R/GATE_OK" ]; do sleep 20; done
  fi
  echo "=== $(date +%T) tier max coding n=21"
  bash "$HERE/run_arm.sh" yamadori coding $RUN 21 0
  TIER_MAX_DONE=1
}

sliced() {  # sliced <arm> <category> <total> <step>
  local n
  for n in $(seq "$4" "$4" "$3") "$3"; do
    maybe_tier_max
    echo "=== $(date +%T) $1 $2 up to n=$n"
    bash "$HERE/run_arm.sh" "$1" "$2" $RUN "$n" 0
  done
}

# This script is the ONLY driver (coordinator, ~10:05: the separately launched
# bonsai coding driver was stopped). It finishes bonsai coding itself; drive.py
# resumes from the answers already on disk, and refuses nothing it already has.
if pgrep -f "[d]rive.py" >/dev/null; then
  echo "another drive.py is running; refusing to start a second driver"; exit 1
fi
sliced bonsai coding 21 2
maybe_tier_max
sliced bonsai+check coding 21 2
sliced minimal coding 21 2
sliced bonsai instruction_following 200 8
while [ "$TIER_MAX_DONE" = 0 ]; do maybe_tier_max; sleep 60; done
echo "=== $(date +%T) single lane: done"
