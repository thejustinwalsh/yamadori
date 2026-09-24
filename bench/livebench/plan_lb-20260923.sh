#!/bin/bash
# Overnight plan for run lb-20260923, as decided by the coordinator at ~02:40:
# BONSAI ARM ONLY. The question LiveBench answers tomorrow is "where does the
# bare model stand against the leaderboard"; stack on vs off comes from the
# domain suite (cheaper tasks, larger n). Measured throughput under tonight's
# shared GPU: ~6.4 tok/s for our one request (4 slots busy) vs ~25 uncontended,
# i.e. ~20 min per coding question -- so coding cannot reach 128 and the stages
# are time-boxed. Reasoning is skipped. The optional exploratory yamadori
# pass (bonsai-wrong + equal random bonsai-right coding questions) runs only
# if both categories complete before ~07:30; at this throughput they will not.
#
# The first coding stage was launched by hand at 01:47 (deadline 03:20); this
# plan resumes coding after it exits. Each stage waits until no drive.py is
# running, so there is never more than one request in flight. A deadline stops
# NEW chunks only; the chunk in flight completes.
#   KEY_FILE=/path/to/key bash plan_lb-20260923.sh
set -u
export TZ=America/New_York
HERE=/mnt/c/Users/jwals/llama-stack/bench/livebench
wait_idle() { while pgrep -f "livebench/drive.py" >/dev/null; do sleep 20; done; }
dl() { local d; d=$(date -d "$1" +%s); [ "$d" -lt "$(date +%s)" ] && d=$(date -d "tomorrow $1" +%s); echo "$d"; }
stage() {
  wait_idle
  echo "=== $(date +%T) stage arm=$1 category=$2 until=$3"
  bash "$HERE/run_arm.sh" "$1" "$2" lb-20260923 0 "$(dl "$3")"
  echo "=== $(date +%T) stage done arm=$1 category=$2"
}
stage bonsai coding                05:40
stage bonsai instruction_following 08:50
echo "=== $(date +%T) plan done"
