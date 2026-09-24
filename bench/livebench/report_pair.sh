#!/bin/bash
# Score everything, then the paired report for one arm against bonsai.
#   bash report_pair.sh <run-id> <arm> [category]
set -u
RUN=$1; ARM=$2; CAT=${3:-coding}
HERE=/mnt/c/Users/jwals/llama-stack/bench/livebench
bash "$HERE/report.sh" "$RUN" | grep -E "^(bonsai|yamadori|minimal|bonsai\+check|   |paired|wrote|->)"
cd "$HERE" && . ~/livebench-run/.venv/bin/activate && export HF_HUB_DISABLE_PROGRESS_BARS=1
python scan_stdin_buffer.py "results/$RUN" 2>&1 | grep -v -i "warn\|hf_token\|Evaluating\|it/s"
python paired_table.py "results/$RUN" "$ARM" "$CAT"
