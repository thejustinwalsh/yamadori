# The dashboard: what it shows, where it reads it, and whether it is current

Audit of 2026-09-29 (operator: "do dashboard fixup please, there is a lot of
drift and a lot of things no longer even working"). Every screen, panel,
`/dash/api/*` route and classic page, checked against the stack as it is
today: live GETs through `:1234` with a test key (read-only; no POST, no
restart), the React build in the browser pane, and the Python that answers.

Where things live (AGENTS.md "Where things live"): the React SPA is `web/`
(committed build `web/dist`, served at `/` by `mcp/dash_static.py`), the JSON
API is `/dash/api/*` (one gated catch-all in `mcp/server.py` asking each
`mcp/dash*.py` module in turn), the Python pages are `/dash/classic*`.

**Restart needed.** The proxy serves `web/dist` from disk, so the rebuilt
SPA is live on a reload. The Python side -- `/dash/api/nebari`, the new
vitals fields (`serving`, slot roles, `vram_line`, the tools-API and SearXNG
probes, the start-time fix), `model` on `/dash/api/tiers`, and the classic
pages' edits -- takes effect when the proxy next restarts. Until then the SPA
says what the running server does not report (NEBARI's panels print
`/dash/api/nebari · HTTP 404`; SERVING prints "serving not reported by this
server"), and nothing is invented in its place.

## 2026-09-30 (later): NEBARI retired, folded into SKILLS

Operator: "Please make that cleanup of Nebari, remove and fold anything
useful into the skills page."

**Moved** to the Skills page's LIBRARY view (`web/src/screens/skills/library.tsx`),
read from the new `GET /dash/api/skill-factory/library` (mcp/dash_skills.py
`library`: read-only, `mode=ro`, no model, cached 60 s, each part fails
alone):

- LIBRARY BY AREA: the SERVED skills (`skills.armed()`, what selection reads)
  counted along framework, language, domain, phase and artifact, with the
  taxonomy's display names, and the counts by state.
- HELD PACKAGES: every `index/packages/*.sqlite3` as package@version,
  embedded or not, complete, items, definitions, files, published -- and
  what reads it today (`PACKAGE_READERS`): the code-search MCP tools on the
  tools API (:1235) -- find_by_pattern, find_definition_opt,
  find_references, read_file_range, and find_by_meaning only when the index
  is embedded (mcp/code_search.py) -- and the skills pipeline's PROVE type
  check, which installs the held version (mcp/typecheck.py `pinned`).
- TIER STRATA (the code-search result tiers over 24 h, `/dash/api/vitals`
  `strata`), below them.

**Removed**: the NEBARI screen and its nav tab (`web/src/screens/Nebari.tsx`,
`web/src/api/nebari.ts`), `/dash/api/nebari` (`mcp/dash_nebari.py`); the
root flare (the recipe-corpus roots' successor: the same served-skill
counts, now LIBRARY BY AREA's bars); RETRIEVAL SOURCES and
`/dash/api/results` with `mcp/dash_results.py` and its suite (NEBARI was the
only reader; `bench/retrieval.py` and its results file stay), the
`KNOWN_VOID` provenance table and `RateBar` (no reader left). `/nebari`
bookmarks are replaced by `/skills` (`router.ts` `LEGACY`,
`useLegacyRedirect`), as `/results` is by `/performance`. The tokonoma's
`tree.nebari` (the cockpit tree's roots) is not this screen and stays.

## 2026-09-30: a view never loads a model; JJAVA; SOKUDO

Operator: "right now the dashboard loading loads bonsai, so we need to make
sure we have a dashboard api that doesn't require that, and we can remove
the old features we don't use anymore. I also need jev/java stats, jev/java
endpoint visibility, graphs of our stats on it ... We should have those
stats built into the skills page too. And we can probably just fully retire
the benchmark page, or repurpose it for just a performance metrics page".

**A view never loads a model.** Any request to llama-swap's
`/upstream/<model>/...` loads that model. Every read a `/dash/api/*` route
reaches, and what changed:

| path | before | now |
|---|---|---|
| `/dash/api/tiers` -> `tiers.accepted_efforts` -> `/upstream/<model>/props` | asked when `/running` did not list the model; **asked when `/running` could not be read** | `/running` unreadable answers the fallback efforts, not kept; `/upstream/<model>/props` only while `/running` lists the model |
| `/dash/api/vitals` -> `vitals.context_pool` -> `budget.pool_size(refresh=True)` | the model's own port, then llama-swap's root `/props` (404 on v256: no load, but it re-read the proxy's pool from any view) | `budget.refresh_direct()` (the main model's own port only) and only while `/running` lists it; else the cached pool, `pool_read` says which; never read at all -> an explained error |
| `/dash/api/vitals/pulse` -> `budget.budgets()` | `pool_size()` when never read (the same two reads) | `vitals.cached_context()`: asks nothing |
| `/dash/api/harness-kit/export/*` -> `catalog.context_window()` | `pool_size()` when never read | only once the process has read the pool (`budget.known_pool()`) |
| `/dash/api/vitals` probes | llama-swap's `/v1/models` (hard-coded port) | the same route at `LLAMA_STACK_URL` (testable) |
| `/dash/api/vitals` `/slots` | the model's own port (never `/upstream`) | unchanged |

Gate: `mcp/test_dash_no_load.py` serves every dashboard GET (the SPA, every
`/dash/api/*` route the SPA and the classic pages read, JJAVA and SOKUDO in
three windows) through `server.app` against a fake llama-swap that records
every request: **zero `/upstream` requests** with the main model off the card
and with `/running` unreadable; with it loaded, `/upstream` only for that
model and only `/props` (the tier ladder's template), the pool read direct.

**Every reader of a model's `/slots`** (the 2026-09-30 03:29:50 stray
`GET /upstream/bonsai/slots`, Python-urllib; logs/stack.log has no
timestamps, so the caller is inferred from its shape: two reads a few
/running polls apart, each loading bonsai): `model.release_slot`,
`skill_questions_bank.slots_busy`, the decider's prime (`decide_turn._idle`,
polled every second while the prime waits) and `decider_bonsai.slot_cells`
were guarded only by `max_mode.blocks()`, which is False when NO main model
is loaded (a GPU window running its own llama-server outside llama-swap);
`bench/decider/bonsai_decider.py` and `bench/octopus/run.py` read
`/upstream/bonsai/slots` unguarded. Each now reads it only while
`gpu_room.model_loaded()` (llama-swap's `/running`, state `ready`) says so,
and otherwise skips and records why (release: `skipped`; slots_busy:
`SLOTS_WHY`, [] not loaded / None unreadable; the prime: not idle; the bench
preflights: `slots_read`, `slots not read`, or Busy when /running cannot be
read). `idle.slots_processing` was already guarded. Gate:
`test_dash_no_load.py` `test_slot_readers_never_load`.

**The bundle is staged, never built in place** (coordinator, 2026-09-30):
the live proxy serves web/dist from disk, so a bundle built there goes live
at once and may call routes the running proxy lacks. `npm run build:stage`
(web/) builds to `web/dist-next` (gitignored); `python
mcp/test_dash_static.py --staged` checks it (YAMADORI_DASH_DIST names any
other); the deploy runs `python scripts/swap_dash_dist.py` right after the
proxy restarts with the matching code (`--check` verifies only,
`--rollback` puts `web/dist-prev` back), then commits web/dist.

**Removed:** the benchmark page -- the React SENTEI screen
(`web/src/screens/Sentei.tsx`, `screens/sentei/*`), the classic
`/dash/classic/results` page and its nav entry, every `/dash/api/results`
section but `retrieval` (NEBARI's RETRIEVAL SOURCES), the LiveBench
electricity estimate (`power.benchmark_estimate`, `benchmark_basis`) and
their types and tests; the HELPER lane count on the tokonoma's SLOTS card
(no job uses the helper lane, docs/REMOVED.md). `/results` bookmarks land on
SOKUDO.

**New pages and routes** (both read-only and model-free, cached 15 s per
window; windows `1h | 6h | 24h | 7d | 30d`, default 24h):

| route | screen | what |
|---|---|---|
| `/dash/api/jjava[/<window>]` (mcp/dash_jjava.py) | `/jjava` JJAVA | decisions per bucket by question set, model and caller (skills injector, skill selection, turn facts, stop judge, Jev `/jev/v1/systemone`, `/v1/systemone`); latency per read, per decision, per burst (a request's decisions, a decider Turn, a Jev call) with p50/p90; each question set's answers, confidence or noul histogram (10 bins), ties, order disagreements, tiers fired; `decide_turn.THRESHOLDS` and `skill_inject.THRESHOLDS`; lane releases ("lane burst ended", kept notes); the Jev API's routes with statuses (401/422/429/529 included), models asked for (jev-* aliases) and served, traffic class, usage tokens, recent calls, the accepted ids and each model's availability now; per-model priors (measured or not); the injector per model |
| `/dash/api/perf[/<window>]` (mcp/dash_perf.py) | `/performance` SOKUDO | decode and prefill tok/s per model per bucket (p10/p50/p90) and by context depth (bins at the gates' depths 4K/8K/32K/64K/128K), roles, tokens; both GPUs' utilisation mean/max, VRAM max, watts, temperature per bucket; the model swaps with load seconds; every gate result file's run sets (Flash-Next gate, Mirai S gate when one exists, bench/results/kv_rank) with n, median, min-max and PASS flags; what was left out and why. KŌGŌSEI (the live 10 minutes) sits beside it |
| Skills · SELECTIONS | the JJAVA · SKILLS INJECTOR panel | stage 1 candidates, stage 2 passed / asked, stage 3's act, injected vs skipped, jjava's failure codes, need / noul histograms, per model |

**Recorded since 2026-09-30** (`mcp/stats_store.py`, `index/stats.sqlite3`
beside the token ledger; a queue put on the hook, one writer thread; only in
a process that called `token_ledger.enable()` -- the proxy, the worker, the
tools API; 30 days kept, the pages' longest window):

| table | hook | what |
|---|---|---|
| generations | `token_ledger.record_upstream` (the proxy's), `token_ledger.record(model=...)` (`model.post`: worker, tools API, decider reads as role `decider`) | model, role, slot, prompt / reused / processed tokens, prefill ms, completion tokens, decode tok/s |
| requests | `recent_turns.note` (x_yamadori) | model, tier, route; the decider Turn (questions, ms, failure, release) and the injector record (numbers only) |
| releases | `slots._note`, `slots.lane_kept_note` | every slot release and lane-kept note |
| swaps | `max_mode.wait_ready` | from, to, load seconds, ok, left loaded |
| gpu | `power.Sampler.sample_once` (the proxy's sampler) | per card per minute: utilisation mean/max, VRAM max, watts mean, temperature max (`vitals.gpus` now reads `temperature.gpu`) |

Also: the decision log's v2 rows carry `ms`, `reads`, `processed_tokens`,
`prompt_tokens` (`decide_turn._log_decision`, additive); the Jev API records
a 401, a body that is not JSON (422) and `GET /jev/v1/models` as `jev_call`
events (`jev_api.refused`, `jev_api.models_call`, called from server.py).

**Left out, and why** (also on the page): tok/s before this deploy (nothing
kept a generation's speed); per-model VRAM (nvidia-smi is per card);
llama-swap's own load times (logs/stack.log has no timestamps); clocks, fan
and PCIe (not sampled); the Mirai S gate until `bench/mirai_s_gate.py`
writes a gate.json.

**Colour** (the dataviz validator, `node validate_palette.js`, on the chart
ground #0a0e17): the categorical slots are the design system's own tokens
in fixed order -- primaryContainer, tertiaryContainer, secondaryContainer,
secondary, then OTHER in outline grey. They sit above the validator's
lightness band (the design system's neon, DESIGN.md), and moss / rose is
7.4 ΔE for a deutan reader (the 6-8 band legal only with secondary
encoding), so every series also has its own dash pattern, a legend beside
the mark and a hover title with every value; the tables carry the numbers.

## Inventory

Status: **keep** (current and working), **fixed** (changed here), **new**
(added here), **retired** (removed from the UI; code left in place or
unrouted, never deleted).

### React screens (`web/src/router.ts`)

| route | screen | reads | status | why |
|---|---|---|---|---|
| `/` | TOKONOMA (cockpit) | vitals, vitals/pulse, power/series, tokens, tiers, mcp, deep, datasets | fixed + new | see the panel rows below |
| `/nebari` | NEBARI | nebari, results, vitals | **retired 2026-09-30** | folded into SKILLS · LIBRARY (by area, held packages, tier strata); `/nebari` redirects to `/skills` |
| `/data` | NAEDOKO | datasets | fixed | the "laya training set" kind is gone from the form (Laya retired 2026-09-24); the recipes kind is labelled "recipes → skills" (extract compiles rows into skills) |
| `/data/:id` | dataset detail | datasets/:id | keep | renders; LABEL/TRAIN still show as OPT stages because `mcp/datasets.py` still defines the Laya branch (see "Not changed") |
| `/results` | SENTEI | results | **retired 2026-09-30** | the benchmark page; `/results` now lands on SOKUDO |
| `/performance` | SOKUDO | perf, power/series | new 2026-09-30 | see "2026-09-30" above |
| `/jjava` | JJAVA | jjava | new 2026-09-30 | see "2026-09-30" above |
| `/settings` | SETTINGS | settings/image | keep | renders both image models and the key's choice |
| `/skills`, `/skills/{create,selections,prompts}` | SKILLS | skills, skill-factory/*, jjava, vitals | keep; + 2026-09-30 panels | all four views render against the live library (552 skills, 511 armed); LIBRARY adds LIBRARY BY AREA, HELD PACKAGES, TIER STRATA (from NEBARI); SELECTIONS adds the JJAVA · SKILLS INJECTOR panel |
| `/skills/:id` | skill detail | skills/:id | keep | renders |
| `/skills/onboarding/:id` | onboarding | skill-factory/onboarding/:id | keep | a missing id is a clean 404 message |
| `/phase0` | the Phase 0 WebGPU gate | none | keep | a design gate, not in the nav; renders |

### Cockpit panels

| panel | source | status | what changed |
|---|---|---|---|
| TOKONOMA (the tree) | vitals, pulse | keep | renders and grows; in a hidden browser pane requestAnimationFrame does not run, so the tree stays a seedling there (not a defect: `window.__bonsai.frames` stops with rAF) |
| SLOTS strip | pulse `slots` | fixed | each slot says its role in the 3-slot layout (`S2 · CHILD`, `· PRIMARY`, `· PINNED`, from `mcp/slots.py` `child_slot()`/`snapshot()`); the card names the model whose server answered; an off-card reason prints without "/slots not answering" in front |
| KV POOL strip and MIKI · KV POOL | vitals/pulse `context` | fixed | the cap layout (2026-09-28) was drawn as "2 CONTEXTS · DEEP THINKING ×1 · RESERVE". Now layout-aware (`web/src/api/kv.ts` `kvNames`): MAIN · VRAM LINE (with where the cap came from), CHILD · DEEP THINKING, SECOND CONVERSATION (host RAM when it spills), and the served `kv_vram_cells` line when reported. The split layout keeps its old names |
| TANE · SEED, SILICON, KŌGŌSEI, DENKI, MIZU, SETSUYAKU | vitals, power/series, tokens | keep | current |
| KEIHŌ · WARNINGS | vitals | keep | now also warns when the tools API or SearXNG does not answer (they are probed) |
| NE · SERVICES | vitals `endpoints`, `listeners` | fixed | watched only llama-swap; the watchdog supervises the tools API (`:1235`) and SearXNG (`:8888`) too. Both are listed and probed (`/health`, `/healthz`: liveness routes that load nothing), in parallel so a down service costs one timeout. Laya (retired) is not watched |
| PROCESSES | vitals `processes` | fixed | `started` read "/Date(1790680397077" (PowerShell 5.1's JSON date, cut at 19 characters); now local ISO seconds, and the SPA reads the old form too |
| SERVING · MAX MODE | vitals `serving` | new | max mode (`mcp/max_mode.py`, docs/FLASH-NEXT.md): which main model holds the 5060 Ti, what llama-swap has loaded (model, port, gguf), in-flight counts, the swap-back rule, which model serves tier `max` |
| header model chip | vitals `serving` | fixed | read the gguf on port 10001 only; the max model gets its own llama-swap port, so in max mode it read "no model on the bonsai port". Now the model on the card, marked `· MAX MODE` when it is the max model |
| DAN · EFFORT LADDER | tiers | fixed | the `max` tier carries "SERVED BY FLASH-NEXT" when max mode is configured (`max_mode.model_for`); features still from `tiers.features` |
| MCP · HOSTED SERVERS | **`/dash/api/mcp`** | new | the endpoint existed with no screen: host on/off, config source, each server's state, package, network, tools (ours ← upstream), licence, call timeout |
| DEEP THINKING · TRIGGERS | **`/dash/api/deep`** | new | the endpoint existed with no screen ("the panel can follow later", 2026-09-24): runs per trigger over 7 days, requests decided, the struggle threshold and its source, per-day labels, the learner's state |

### `/dash/api/*` (GET unless noted; all behind the key)

Live status: every GET below answered 200 on 2026-09-29 (times from one read
each, n=1).

| route | module | status | notes |
|---|---|---|---|
| `vitals` | dash_vitals → vitals.snapshot | fixed | + `serving`, slot `role`/`pinned`/`primary`, slots `model`, `context.vram_line`, tools-api/searxng listeners and probes, start-time parse. /slots is read from the max model's port while it holds the card |
| `vitals/pulse` | dash_vitals → vitals.pulse | fixed | the same slot and context fields; no new I/O unless max mode is configured (then llama-swap's `/running`, cached 2 s) |
| `power`, `power/series` | dash_vitals → power | keep | |
| `tiers` | dashboard | fixed | + per-tier `model`, + `max_mode` |
| `nebari` | dash_nebari | **retired 2026-09-30** | its skills-by-area and held-packages parts moved to `skill-factory/library` (dash_skills); the module is gone |
| `stats` | dashboard | retired from the SPA | the recipe corpus's counts; still answers (the classic corpus page's numbers) |
| `recipes`, POST `recipe` | dashboard | keep (classic only) | edits bench/recipes rows, which are provenance only since the corpus became skills; the classic page now says so |
| `results` | dash_results | **retired 2026-09-30** | the benchmark sections went with the page, the retrieval section with NEBARI (its only reader); the module is gone |
| `jjava`, `jjava/<window>` | **dash_jjava (new)** | new 2026-09-30 | read-only, model-free |
| `perf`, `perf/<window>` | **dash_perf (new)** | new 2026-09-30 | read-only, model-free |
| `datasets`, `datasets/:id`, POST `dataset`, `dataset/{answer,advance,rerun,assist}` | dash_data | keep | POSTs not exercised (read-only audit) |
| `skills`, `skills/:id`, `skills/:id/skill.md`, `skill-factory/{prompts,selections,recent,onboarding,onboarding/:id,library}` | dash_skills | keep; `library` new 2026-09-30 | `skills` takes ~3.5 s (1.3 MB); POSTs not exercised; `library` is NEBARI's useful part (by area, held packages) |
| `tokens` | dash_tokens | keep | |
| `deep`, POST `deep/revert`, `deep/e1/{revert,decide}` | dash_deep | keep | now drawn by the cockpit |
| `mcp` | dash_mcp | keep | now drawn by the cockpit |
| GET/PUT `settings/image` | server.py | keep | |

### Classic pages (`/dash/classic*`, the Python fallback)

| page | status | notes |
|---|---|---|
| `/dash/classic` (recipe corpus review) | fixed | says the rows are not served and an edit changes the row, not an armed skill |
| `/dash/classic/data` | fixed | the "laya training set" kind is gone |
| `/dash/classic/vitals` | fixed | the context pool reads the cap layout (main = the VRAM line, child, second conversation); the "recipe corpus" link is gone from its lede |
| `/dash/classic/results` | **retired 2026-09-30** | the classic benchmark page, with its nav entry |

Each classic page's script was checked with `node --check` after the edits.

## Not changed here -- someone else's call or someone else's files

- **Layout v2** (in preparation in `mcp/proxy.py`, `mcp/tiers.py`,
  `mcp/vision.py`, `mcp/budget.py`, `mcp/slots.py`; not touched). The
  dashboard names the child slot "CHILD · DEEP THINKING" and describes it as
  "deep thinking, the decider and side calls" (`web/src/api/kv.ts`
  `kvNames`), from AGENTS.md as it stands. When the child becomes a small
  decider lane and the second brain is off by default, that text should
  follow; the cleanest source is a field of `budget.budgets()` naming the
  child's role (e.g. `helper_role`), which the panel would print instead of
  its own words. Vision moving to an on-demand A4000 model needs nothing
  here: SERVING lists whatever llama-swap has loaded, and SILICON the card.
  The tier ladder reads `tiers.features` live, so a changed matrix shows
  without a dashboard change.
- `mcp/datasets.py` still has the `laya` kind and the LABEL/TRAIN stages
  (`dataset.train` runs `scripts/train_laya.py`). The forms no longer offer
  the kind; retiring it and the two stages in the backend is the dataset
  pipeline's change.
- The recipe review surface (`/dash/classic`, `/dash/api/recipes`, POST
  `/dash/api/recipe`) still edits bench/recipes, which nothing serves.
  Unrouting it is the operator's decision; it is labelled, not removed.

## Checks

- Offline: `mcp/test_vitals.py` (74; the new checks: start-time parse,
  watched services and parallel probes, serving, max-mode /slots and the
  off-card reason, slot roles, the VRAM line), `mcp/test_dash_now.py` (24:
  /dash/api/nebari, tiers' model, the server route), `mcp/test_dash_static.py`
  (49: the rebuilt bundle is current), plus the suites that read these
  modules. Run under `scripts/run_tests.py -k` (the offline guard).
- Web: `npx vitest run` in `web/` (254), `tsc --noEmit`, and the build with
  Node 24.21.0 (`web/.npmrc` is engine-strict; Node 20.15 cannot build).
- Live: every GET above through `:1234` (200). The rebuilt SPA against the
  running proxy, and against a local GET-only relay that adds the not-yet-
  served fields from the same code (`dash_nebari.overview()` over a read-only
  copy of the jobs database), in the browser pane at 1600 px and 375 px (no
  horizontal overflow).
