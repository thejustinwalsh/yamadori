#!/bin/bash
# Launch one arm x category from WSL. The key is read from KEY_FILE by drive.py
# into the child environment only; it is never echoed or written.
#   KEY_FILE=/mnt/c/.../dogfood.key bash run_arm.sh <arm> <category> <run-id> [n] [deadline-epoch] [paired-with-arm]
set -euo pipefail
ARM=$1; CAT=$2; RUN=$3; N=${4:-0}; DEADLINE=${5:-0}; PAIRED=${6:-}
HERE=/mnt/c/Users/jwals/llama-stack/bench/livebench
. ~/livebench-run/.venv/bin/activate
export LB_TIMEOUT=${LB_TIMEOUT:-7200}
export HF_HUB_DISABLE_PROGRESS_BARS=1
mkdir -p "$HERE/results/$RUN/logs"
python -u "$HERE/drive.py" --arm "$ARM" --category "$CAT" --run-dir "$HERE/results/$RUN" \
  --n "$N" --deadline "$DEADLINE" --key-file "${KEY_FILE:?set KEY_FILE}" \
  --api-base "${LB_API_BASE:-http://172.29.32.1:1234/v1}" ${PAIRED:+--paired-with "$PAIRED"} \
  ${LB_PARALLEL_FILE:+--parallel-file "$LB_PARALLEL_FILE"} \
  2>&1 | tee -a "$HERE/results/$RUN/logs/progress_${ARM}_${CAT}.log"
