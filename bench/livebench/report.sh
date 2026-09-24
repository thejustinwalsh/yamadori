#!/bin/bash
# Score everything answered so far with the official judge, then summarize and compare.
#   bash report.sh <run-id>
# CPU only (no model calls), safe to run while an arm is generating.
set -u
RUN=${1:-lb-20260923}
HERE=/mnt/c/Users/jwals/llama-stack/bench/livebench
cd "$HERE"
. ~/livebench-run/.venv/bin/activate
export HF_HUB_DISABLE_PROGRESS_BARS=1
for arm in $(python -c "import drive; print(' '.join(a for a in drive.ARMS if not a.endswith('_prefix')))"); do
  disp=$(python -c "import drive; print(drive.ARMS['$arm']['display'])")
  for cat in coding reasoning instruction_following math data_analysis language; do
    if ls ~/livebench-run/LiveBench/livebench/data/live_bench/$cat/*/model_answer/$disp.jsonl >/dev/null 2>&1; then
      python score.py --run-dir "results/$RUN" --arm "$arm" --category "$cat"
    fi
  done
done
python summarize.py "results/$RUN" 2000
python compare.py "results/$RUN" reference/leaderboard_2026-09-23.json reference/table_2024_11_25.csv 2>&1 | grep -v -i "warn\|hf_token"
