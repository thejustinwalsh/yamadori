# Removed 2026-09-29

The operator, 2026-09-29: "these old things are failed decisions, they are
documented, they are not kept around for later. It should be in GitHub if we
want to go back, clean it all up."

So the features below were **deleted**, not switched off: the code, its
switches and tier flags, its tests, its dashboard panels and its
`x_yamadori` fields. **The way back is commit `e360d37`** ("Checkpoint before
removing the failed second-brain machinery", pushed to GitHub before the
removal): `git show e360d37:mcp/deep.py`, or check out that commit to see the
whole system as it ran.

The product stays a proxy: `mcp/proxy.py` was edited, not deleted (8,851 ->
~5,970 lines). About 11k lines of Laya/CLM code and harnesses and ~8k lines of
second-brain code went with their suites, bench harnesses and data.

## What went, and why

| what | where it lived | why it went |
|---|---|---|
| **Deep thinking** -- the second brain's `investigate` and `plan` jobs, the four-section hand-off and plan, THE EVIDENCE, the machine-built hand-off | `mcp/shomen.py` (2,175 lines), `proxy._deep_thinking` | The model never used it on its own: `yama_think_deeply`, offered for whole runs, was called 0 times voluntarily (`bench/mcp/lookup_probe.jsonl`, Pi, n=4 valid trials). Pushed by the proxy it made runs worse: on the pagoda runs (2026-09-28) the probe triggers ran deep thinking for 14-18 minutes a package, and hand-offs were echoed or continued as prose instead of acted on (pagoda-h2: finish=stop with no call after the kickoff plan). Nothing measured a gain. |
| `yama_think_deeply`, `yama_plan`, the kickoff plan hop (`synthetic_hop`) | `mcp/deep.py`, `proxy._think_deeply`, `_plan_task` | as above |
| The triggers: struggle, known-hard area, kickoff, and the server-tool triggers (probe, scratch, next_piece, plan_done, implement) | `mcp/deep.py` (1,977 lines) | as above; every threshold was "a CHOICE, unmeasured" (docs/CONSTANTS-AUDIT.md) |
| The verify directive | `mcp/verify_moment.py` | fired by the same triggers; never run live |
| continue_stated_step | `proxy._stated_step*`, `CONTINUE_LINE` | an approved fix for one observed shape, never run live |
| `deep_decisions` and the idle-time learner | `mcp/deep.py`, `mcp/deep_learn.py`, `mcp/dash_deep.py` (`/dash/api/deep`) | nothing was ever learned from them (`deep_adjustments` was empty when checked, 2026-09-26) |
| The research tools only deep thinking used: `find_in_knowledge_base`, `read_web_page`, `search_web` | `mcp/research_tools.py` | only the second brain called them. Its pinned GET stays as **`mcp/pinned_fetch.py`**: the MCP host reads READMEs and npm packuments through it. SearXNG stays configured (start-stack, watchdog); nothing in the proxy searches the web now. |
| **Fan-out** (candidates B and C, the tie-breaker, the hand-back) | `mcp/fanout.py` (1,470 lines), `proxy._fan_out`, `_finish_text` | its one measurement was a null: 7/8 vs 7/8, zero discordant pairs, at 3.2x wall clock (n=8) |
| **The fix-up repair** and **the code-check notes** ("Verified", "Repaired", "Checked"), the answer repair, the formatter check | `tool_code` (check, fixup, notes), `code_check` (review_answer, formatter, lint, style, the check_code tool surface), `tools/format/` | the model imitated the notes in its own content (#36: 13-27 imitated lines a run); no result showed the repair helped. **Kept:** `tool_code`'s detection (the skills, the no-progress guard and result compression read it) and the **image guard**; `code_check`'s parse layer (the router, the skills' proof). |
| **The addendum** (`proxy.ADDENDUM`, `ADDENDUM_THINK_ROW`, `ADDENDUM_TOOLS`) | `mcp/proxy.py` | it described the second model's fold-backs and tools, which are gone |
| **The delegate arm** (`delegate_investigation`, `YAMADORI_DELEGATE_TOOL`) | `mcp/proxy.py` | a benchmark arm of deep thinking |
| **Laya** -- the service, its heads, trainer, calibration, judge, the `laya` dataset kind and `train` stage, its harnesses | `mcp/laya_*.py`, `mcp/session.py`, `mcp/dialectic.py`, `scripts/train_laya.py`, `calibrate_laya.py`, `eval_judge.py`, `ws_check.py`, `bench/laya_*`, `bench/tev1/` | E1 beat it (route_in 104/120 vs 80/120, p=0.0001; docs/E1.md) and it was already off the request path (retired 2026-09-24). E1's training labels (`bench/laya_routing_labels*.jsonl`, the held-out set) are kept. |
| **CLM** -- the client, heads, the decider, its bench | `mcp/clm.py`, `mcp/clm_heads.py`, `skill_deciders.ClmDecider`, `bench/clm/` | retired 2026-09-28 for the Bonsai decider; never the default |
| **Selection's legacy path** (the regex + symbol lookup + Laya/E1 two-signal decision, disagreement escalating) and selection's investigate / fan-out decisions | `mcp/selection.py` | it decided deep thinking and fan-out, both gone; held-out 91/120 against 89/120 for the rule alone (p=0.79, not significant). **Kept:** the skills decision, the utility (side-call) rule, the question readers, `rule_baseline` (E1's comparison), the symbol lookup the router reads. |
| **Library definitions** (a library question's held-symbol definitions) | `proxy._library_definitions` | replaced by the MCP host's package lookups (operator decision (4), 2026-09-29). |
| **LIBRARY USE** (#19: the definitions of names a conversation imports from a held package, appended to the message a request ended on) | `proxy._library_use`, `library_uses` | Decided here: removed with the definitions. Same mechanism (`find_definition_opt` on the held package indexes, pushed into the context unasked), an UNMEASURED choice from the start, and the pagoda evidence (2026-09-28) is that what we push into the context unasked makes runs worse while exact names on the model's own call made the run work; the MCP lookups answer on the model's call (THE TOOL RECIPE). A tool result that already carries one replays it from the ledger byte for byte. |
| `progress.learn` / `project_hint` (the per-step project inference) | `mcp/progress.py` | served deep thinking's label rule 2 (#52). `project_of` / `is_project` stay for the skills. |
| Switches and tier flags | `tiers.BEHAVIOURS`: `seed_frame`, `helped_needs_change`, `plan_prompt`, `deep_tool_hop`, `continue_stated_step`, `auto_triggers`, `verify_directive`; `tiers.TIERS`: `retrieval`, `fanout`, `investigate`, `check_code`, `repair`, `delegate`; `JOB_THINKING` rows, `HELPER_NUDGE_MESSAGE` | the features they switched are gone. `tiers.from_header` drops the old flags silently, so an old benchmark header still parses. |
| `x_yamadori` fields | `fanout`, `investigate`, `deep`, `continued`, `check_code`, `repair`, `tool_code`, `fold_back`, `fold_back_answer`, `library_use`, `tools_gate`; `selection.investigate` / `fanout_n`; `progress.project` | the features that filled them are gone |
| Dashboard panels | the DEEP THINKING · TRIGGERS panel, the fan-out channel of the bonsai view, the UNSEEN chip, the Laya dataset stage | as above; `web/dist` rebuilt |

**2026-09-30, the dashboard** (operator: "we can probably just fully retire
the benchmark page"): the benchmark page -- React SENTEI, the classic
`/dash/classic/results`, every `/dash/api/results` section but `retrieval`
(NEBARI), the LiveBench electricity estimate (`power.benchmark_estimate`,
`benchmark_basis`) -- and the tokonoma's HELPER lane count. SOKUDO
(`/performance`) replaces the page. Later that day (operator: "Please make
that cleanup of Nebari, remove and fold anything useful into the skills
page"): the NEBARI screen, `/dash/api/nebari` (`mcp/dash_nebari.py`), its
root flare and RETRIEVAL SOURCES, and `/dash/api/results`
(`mcp/dash_results.py`, its last reader gone); its skills-by-area and held
packages moved to the Skills page. docs/DASHBOARD.md "2026-09-30".

## What stayed

- **rings** (the work log the proxy writes and re-injects after a compaction).
- **The whole skills system** (pipeline and serving; skills are off at every
  tier, operator 2026-09-29, and the code stays), **E1** (held for the jjava
  rewrite), the decider files.
- **`proxy.directive_prefill`** and the ledger's restore of prefilled reasoning
  (operator: kept as a skill delivery channel the skills renderer measures).
- The MCP host, images, vision, the ledger, sessions, compaction, slots and
  the lane, max mode and the tier -> model table, the Responses and Messages
  APIs, `mcp/progress_guard.py`, `summarize_text` (the MCP tool, through
  `mcp/model.py`).
- The helper-lane infrastructure (`admission.helper_lane`, the helper share
  in `mcp/budget.py`, `slots.helper_slot`) is still there, unused by any job:
  it is part of the KV layout, which is another decision's.

## What moved

**The concept seed.** It rode in every second-brain job's user message; now it
is drawn once per conversation (`concept_seed.seed_for`, via
`proxy._draw_seed`) and appended to the conversation's **first user turn** as
`concept_seed.USER_TURN_LINE`, part of that turn's ledger-recorded injection
(`proxy.INJECT_PARTS` = skills, seed, work log), so it replays byte for byte
and a compaction served on the ledger's rendering summarises it with the
rest. Tiers `high`, `xhigh`, `max` (tier flag `seed`; the tiers where the
second brain drew one); `X-Yamadori-Features {"seed": false}` turns it off.
Recorded by the conversation's lineage and reported as
`x_yamadori.session.seed` {word, token_id, u32}.

## Deploy steps (not done here: config and launch files are edited at deploy)

1. Restart the proxy, the tools API and the worker (the code changed), then
   `python scripts/deploy_check.py --key-file PATH` must exit 0.
2. `config.yaml` / `config.template.yaml`: remove the `clm-encoder` model and
   its group entry, and at the same time the `clm-encoder` row of
   `mcp/gpu_room.py` `SIZES` (`mcp/test_gpu_room.py` checks the two
   against each other).
3. `models/manifest.yaml`: retire `qwen3-8b-clm-q8`, `qwen3-8b-clm-bf16`, the
   Qwen3-8B safetensors inputs, `clm-v0.1-8b-pt`, `clm-heads-v0.1`,
   `clm-heads-skills-v1` and the `laya-*` entries; keep
   `qwen3-8b-tokenizer` with its role changed to
   `skill_packages.WORDS_TOKENIZER` (the English-word lexicon).
4. `scripts/start-stack.bat` / `scripts/watchdog.ps1`: nothing required --
   Laya is already marked retired there and not started. SearXNG stays.
   Environment variables nothing reads any more (harmless if set):
   `YAMADORI_DELEGATE_TOOL`, `YAMADORI_SEED_FRAME`,
   `YAMADORI_DEEP_HELPED_NEEDS_CHANGE`, `YAMADORI_DEEP_TOOL_HOP`,
   `YAMADORI_AUTO_TRIGGERS`, `YAMADORI_VERIFY_DIRECTIVE`,
   `YAMADORI_CONTINUE_STATED_STEP`, `YAMADORI_PLAN_PROMPT_V2`,
   `YAMADORI_PLAN_THINKING`, `YAMADORI_HELPER_NUDGE`,
   `YAMADORI_REPAIR_ROUNDS`, `YAMADORI_FORMAT_DIR`,
   `YAMADORI_FORMAT_TIMEOUT`, `YAMADORI_SEARCHES_PER_RUN`,
   `YAMADORI_READS_PER_RUN`, `YAMADORI_SEARCH_BACKOFF_S`,
   `YAMADORI_STRUGGLE_THRESHOLD`, `YAMADORI_UNSEEN_PACKAGES`,
   `YAMADORI_MODEL_CUTOFF`, `LAYA_URL`, `YAMADORI_LAYA_ROUTE_TIMEOUT`
   (`YAMADORI_SEARCH_URL` is still set by the launchers for SearXNG).
5. Optional cleanup of untracked state: `index/laya/`, `index/clm/`,
   `index/clm_actions.npz`, `index/e1/heads/escalate/`, `.venv-laya/`,
   `locks/laya.lock.txt`, `tools/format/node_modules/`. The
   `deep_decisions` table in `index/corpus.sqlite3` is no longer written or
   read. Queued `deep.learn`, `dataset.label` or `dataset.train` jobs in
   `index/jobs.sqlite3` have no handler any more.
6. Commit `web/dist` with `git add -A web/dist` (the rebuild renamed its
   hashed assets).

# Removed 2026-10-01: the reranker

The operator, 2026-10-01, verbatim: "Remove reranker". This supersedes
AGENTS.md's "The reranker is not cut, not trusted, and not used" (the hold
that waited for its rank path to be fixed and re-measured). Deleted, not
switched off, like the removal above; **the way back is commit `e360d37`**
(`git show e360d37:mcp/code_search.py`, `git show
e360d37:config.template.yaml`; config.yaml itself is untracked).

**What it was.** llama-swap model `reranker`: Qwen3-Reranker-0.6B-Q8_0
(mradermacher's GGUF, `models/manifest.yaml` `qwen3-reranker-0.6b-q8`) on
`llama-prism`, `-c 16384 --reranking --pooling rank`, on the A4000 in the
`retrieval` group beside `embeddings`, preloaded at startup and resident.

**Why.** Nothing on the request path trusted it. `docs/FINDINGS.md` #20:
`/v1/rerank` scores depend on batch composition -- 89 documents, each
queried with its own verbatim text, scored 15-16/89 batched against 68/89
one at a time (embeddings 89/89); the live gate of 2026-09-24 read 16/89
batched vs 66/89 solo (`bench/test_hint_collapse.py --live`, kept as an
expected failure in `scripts/run_tests.py` until today). That voided every
reranker number in the repo in both directions. In use it reordered only
`code_search.search` at `top_k <= RERANK_MAX_K` (2): find_by_meaning
(`search_fused`, which asks for at least 10) never reached it, and the tools
API's `POST /search` did only when a caller asked for 1 or 2 results.

**The room it frees.** ~2.9 GB of the A4000, resident since startup (the
coordinator's reading when it unloaded the model from the running
llama-swap, 2026-10-01; gpu_room's `SIZES` row was an ESTIMATE of 3,000 MiB,
never measured alone). Numbers measured with it resident are kept as
measured and say so: `bonsai-a4000`'s fit (`-c 141,312`, bench/a4000_fit.py,
2026-09-30; the 4,152 MiB of residents in its arithmetic included the
reranker) and the 7,565 MiB retrieval + Laya figure.

| what | where it lived | now |
|---|---|---|
| The model entry, its `retrieval` group membership and its `hooks.on_startup.preload` line | `config.yaml`, `config.template.yaml` | removed; `retrieval` keeps `embeddings` |
| `code_search.rerank()`, `RERANK_MODEL`, `RERANK_DOC_CHARS`, `RERANK_MAX_K` and the rerank branch of `search()` | `mcp/code_search.py` | removed; `search()` returns embedding (cosine) order, `score` = `sim` |
| The `yamadori-rerank` catalog alias | `mcp/catalog.py` | removed |
| The `reranker` row of `SIZES`, and its place in `RESIDENT` | `mcp/gpu_room.py` | removed (`mcp/test_gpu_room.py` checks SIZES against config.yaml) |
| The reranker's checks | `mcp/test_gpu_room.py` (fake `/v1/rerank`, eviction order), `mcp/test_catalog.py`, `bench/test_hint_collapse.py` (the rerank floors, the McNemar and degeneracy checks, the whole `--live` arm), `scripts/run_tests.py` `EXPECTED_FAIL` (now empty) | removed; test_gpu_room's eviction-order check now evicts the embedder under a 1,500 MiB foreign load |
| `scripts/eval_rerank.py` | -- | deleted (reranker only) |
| The `rerank` / `rerank_batched` arms | `bench/hint_collapse.py`, `bench/data/hint_collapse_runs.json` (their cached scores) | removed; the other arms replay unchanged |
| The `rerank` selector | `bench/mechanisms/selectors.py` (`select_rerank`, its hook, `mock_rerank`, dry-run checks 7 and 8) | removed |
| The `embed+rerank` condition | `bench/retrieval.py` | removed; `bench/retrieval_results.jsonl` keeps its `rerank_rank` / `rerank_ms` columns as the record (void per #20) |
| The model file's manifest entry | `models/manifest.yaml` | `status: retired`, no `config_entries`; `engines/manifest.yaml` `llama-prism` `config_refs` no longer lists it |

**Kept, as history or data:** docs/FINDINGS.md, PLAN, ROADMAP, SELECTION,
HINTS, research/*; `bench/retrieval_results.jsonl` (its embedding and symbol
columns are cited); `bench/semantic_tasks.py` (prose locate questions, for
any retriever); the guardrail label texts that mention a reranker
(`bench/guardrail_labels.jsonl` and `bench/guardrail_labels_build.py`,
labelled data); the captured
`web/src/bonsai/__fixtures__/vitals.live.json`.

**Deploy steps (not done here):** config.yaml is edited, but llama-swap
re-reads it only on a restart; the coordinator already unloaded the model
from the running llama-swap. At the next restart the reranker is not
preloaded. Restart the tools API and the proxy for the code change. The
model file `${models}/Qwen3-Reranker-0.6B-Q8_0.gguf` may be deleted by the
operator (the manifest keeps its sha256 and source to re-fetch it).
`RERANK_MODEL` is an environment variable nothing reads any more.
