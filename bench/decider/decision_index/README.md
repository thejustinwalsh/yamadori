# Decision Index (v0.2.1) on jjava

The community harness (https://github.com/apolinario/decision-index, commit `87d4650b42b377c0291a89c1f1a879f9b31082bf`,
"Version 0.2.1") run against OUR Jev API (`<proxy>/jev/v1/systemone`, models `jjava-latest` ...), through the harness's own
`http` engine (a subclass, `jjava_engine.py`). No data lives in this directory; the harness checkout, the rebuilt suite and
the runs are under `C:\Users\jwals\octo\decision-index` (outside the repo).

Operator, 2026-10-06: "download the benchmark data set and let's run ours". Facts as of 2026-10-06 (see the report that
came with this commit for the measurements; this file records how, not results):

- **The suite is not downloadable.** The kit's `suite download` reads the Hub dataset `multimodalart/decision-index-suite-0.2`,
  which answers 401 without a login (the README says to build your own private copy). The suite is REBUILT from the pinned
  public sources (`suite rebuild`: ~6.9 GB of downloads, `acquire_nohle.py`, then `build_nohle.py`).
- **HLE (catalog 45, 513 rows, 501 scoreable) could not be fetched**: `cais/hle` is gated on the Hub (accept the terms + log in;
  an operator action). The rebuilt corpus therefore lacks it: 124,458 of 124,971 base rows, so the corpus sha256 is NOT the
  pinned `b2b56d6f...` (it is `6787a1d5...`). Everything else was verified byte-for-byte: the 7 added benchmarks'
  `added-rows.jsonl` equals the pinned `7429f3c9...`; every one of the 42 other benchmarks' normalized source files equals the
  sha256 in `hub/manifest.json`; row counts equal the manifest's; ToolRet/BRIGHT/Home appliances match the 0.2.1 subset lists.
  A run over this corpus is leaderboard-comparable only on the other 43 benchmarks, never as a full-suite index.
- **Windows needs shims** (`build_nohle.py`): the harness was written for Linux. Text files are written with `\n` (not
  CRLF), provenance paths are POSIX, API-Bank's file order is case-sensitive (Windows' `Path` sort is not), and
  `PYTHONUTF8=1` is required (cp1252 default breaks the build and reading the 0.2.1 subset JSON, which holds a U+2019).
  Clone with `core.autocrlf=false` (the CRLF checkout changes `decision_index/data/**` and fails `pytest`'s hash pins), and
  fetch git sources with `GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=core.autocrlf GIT_CONFIG_VALUE_0=false`.
- `suite import --no-verify` stages it as `suite-0.2/`; `score` works on partial runs (`complete: false`).

Files: `jjava_engine.py` (waits Retry-After on 429/529, 422 `too_long` -> `Unsupported`; everything else is the harness's
http engine: payload and response untouched), `test_jjava_engine.py` (offline, fake server), `make_samples.py` (the 86-row
compatibility list and a reconstruction of the 750-row latency sample; the lab's own files are not in the kit),
`run_pass.ps1`, `summarize_results.py`, `requirements.lock.txt`.

The runner is sequential (one request at a time); it sends two warm-up requests ("The color is red.") at start.
