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

## Inventory

Status: **keep** (current and working), **fixed** (changed here), **new**
(added here), **retired** (removed from the UI; code left in place or
unrouted, never deleted).

### React screens (`web/src/router.ts`)

| route | screen | reads | status | why |
|---|---|---|---|---|
| `/` | TOKONOMA (cockpit) | vitals, vitals/pulse, power/series, tokens, tiers, mcp, deep, datasets | fixed + new | see the panel rows below |
| `/nebari` | NEBARI | **`/dash/api/nebari`** (was `/dash/api/stats`), results, vitals | fixed | its roots were the recipe corpus's domain tags; the corpus is not served since 2026-09-26 (skills). Now: served skills by framework/language (the flare), SKILLS (counts by state, served by domain and phase), HELD PACKAGES (every index, UNSEEN per `deep.unseen`), RETRIEVAL SOURCES (measured history, kept), TIER STRATA (kept) |
| `/data` | NAEDOKO | datasets | fixed | the "laya training set" kind is gone from the form (Laya retired 2026-09-24); the recipes kind is labelled "recipes → skills" (extract compiles rows into skills) |
| `/data/:id` | dataset detail | datasets/:id | keep | renders; LABEL/TRAIN still show as OPT stages because `mcp/datasets.py` still defines the Laya branch (see "Not changed") |
| `/results` | SENTEI | results | keep | historical measurements; "hints" rows are the arms as they ran then, not the current stack. The effort-tier table is live from `tiers.features` |
| `/settings` | SETTINGS | settings/image | keep | renders both image models and the key's choice |
| `/skills`, `/skills/{create,selections,prompts}` | SKILLS | skills, skill-factory/* | keep | all four views render against the live library (552 skills, 511 armed) |
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
| `nebari` | **dash_nebari (new)** | new | skills counts, served skills by taxonomy axis, held package indexes, code/repo index counts; read-only, cached 60 s. Registered in server.py's GET chain |
| `stats` | dashboard | retired from the SPA | the recipe corpus's counts; still answers (the classic corpus page's numbers) |
| `recipes`, POST `recipe` | dashboard | keep (classic only) | edits bench/recipes rows, which are provenance only since the corpus became skills; the classic page now says so |
| `results` | dash_results | keep | |
| `datasets`, `datasets/:id`, POST `dataset`, `dataset/{answer,advance,rerun,assist}` | dash_data | keep | POSTs not exercised (read-only audit) |
| `skills`, `skills/:id`, `skills/:id/skill.md`, `skill-factory/{prompts,selections,recent,onboarding,onboarding/:id}` | dash_skills | keep | `skills` takes ~3.5 s (1.3 MB); POSTs not exercised |
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
| `/dash/classic/results` | keep | |

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
