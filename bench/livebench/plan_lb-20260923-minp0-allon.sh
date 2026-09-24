#!/bin/bash
# lb-20260923-minp0 after the ~11:25 deploy (tier xhigh, 10-turn tool cap). Operator: all-on vs
# all-off data today. ONE request in flight (the proxy enforces MAIN_LANES=1).
#   1. yamadori-xhigh coding: gate n=1 -> wait for GATE_OK_XHIGH (created by hand after
#      inspection) -> all 21  [the all-on arm, effort-matched to bonsai bare @ medium]
#   2. HOLD. Tier max is OFF the queue (operator, ~11:15: focus only on tier xhigh). It stays a
#      partial arm: the gate row plus one not_run (runaway xhigh thinking, killed at deploy).
# Refuses to start if any other driver is alive.
#   KEY_FILE=/path/to/key bash plan_lb-20260923-minp0-allon.sh
set -u
HERE=/mnt/c/Users/jwals/llama-stack/bench/livebench
RUN=lb-20260923-minp0
R=$HERE/results/$RUN
if pgrep -f "[d]rive.py|[g]en_api_answer.py" >/dev/null; then
  echo "another driver or harness is running; refusing to start"; exit 1
fi
echo "=== $(date +%T) yamadori-xhigh gate (n=1)"
bash "$HERE/run_arm.sh" yamadori-xhigh coding $RUN 1 0
echo "=== $(date +%T) yamadori-xhigh gate answered; waiting for GATE_OK_XHIGH"
while [ ! -f "$R/GATE_OK_XHIGH" ]; do sleep 20; done
echo "=== $(date +%T) yamadori-xhigh coding n=21"
bash "$HERE/run_arm.sh" yamadori-xhigh coding $RUN 21 0
echo "=== $(date +%T) HOLD: yamadori-xhigh done; tier max is off the queue"
