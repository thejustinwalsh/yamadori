# bench/swebench: SWE-bench Verified, the public way

This directory runs the benchmark that public model comparisons use:
SWE-bench Verified, driven by mini-swe-agent with the bash-only leaderboard
config, and graded by the swebench harness in Docker. The model is ours,
reached through the proxy on `:1234`.

For the standard itself, the sources, the published scores and the pilot
numbers, see [docs/SWE-BENCH.md](../../docs/SWE-BENCH.md).

| file | what it does |
|---|---|
| `arms.py` | Defines the three arms (`bonsai`, `yamadori-auto`, `yamadori`) and their `X-Yamadori-Features` headers. Also holds the pinned versions, the datasets and the pilot ids. |
| `run.py` | The Windows side and the entry point. It waits for the card and holds the worker's gpu lane. It runs each instance through WSL, evaluates, and writes `results.jsonl`. |
| `wsl_side.py` | The WSL side. It runs `check`, `ids`, `agent` (one instance with mini-swe-agent) and `evaluate` (the swebench harness). |
| `dockerfix.py` | Runs mini-swe-agent and the harness natively on Windows, around Docker Desktop's broken WSL integration and stale inspect cache (2026-09-23). This is the default path; `SWEBENCH_SIDE=wsl` switches back. It also retries a 429 after 1–2 s. `dockerfix.py selftest` proves the environment matches the stock one. |
| `parse_results.py` | Turns the trajectories and the harness reports into `results/<run-id>/results.jsonl`. |
| `test_parse_results.py` | Proves the parser keeps resolved, unresolved, empty-patch and agent-error apart. It runs in `scripts/run_tests.py`. |

## One-time setup (WSL Ubuntu)

```bash
mkdir -p ~/swebench-yamadori && cd ~/swebench-yamadori
uv venv --python 3.11 .venv && . .venv/bin/activate
uv pip install "mini-swe-agent==2.1.0" "swebench==4.1.0"
python /mnt/c/Users/jwals/llama-stack/bench/swebench/wsl_side.py check
```

`check` prints the installed versions and the sha256 of the leaderboard config
that ships in the package. It must be
`4f0492bb3dd77e21b287014eb077f22ac4a932a46087cf82f053a195f3e7778c`, the same
file as the v2.0.0 and v2.1.0 git tags.

## Running

Run these from the repo root with the stack interpreter. `K` is the path to a
file that contains the API key. The key is never passed as a value.

```powershell
$PY = "C:\Users\jwals\textgen\installer_files\env\python.exe"
& $PY bench/swebench/run.py --run-id pilot-20260922 --arms bonsai,yamadori-auto,yamadori --pilot --key-file K
& $PY bench/swebench/run.py --run-id mini50-bonsai --arms bonsai --subset verified-mini --all --key-file K
& $PY bench/swebench/run.py --run-id verified-bonsai --arms bonsai --all --key-file K
```

Runs are resumable. To continue one, re-run the same command: any instance
already in an arm's `preds.json` is skipped. To regrade without the model, use
`--evaluate-only`.

## Output

Everything for a run goes under `results/<run-id>/<arm>/`:

- `preds.json`
- `<instance>/<instance>.traj.json`: the full conversation, including every
  raw response with `usage` and `x_yamadori`
- `timings.jsonl`
- `run.json`: versions, config hash, the overlay and every deviation
- `mini_stdout.log`
- the harness report and `logs/`

`results/<run-id>/results.jsonl` holds one row per (instance, arm).
