# Package onboarding: from a prompt with links to a held, tested package

DESIGN, 2026-09-27; **BUILT the same day, offline** (section 0). No GPU ran
and nothing was sent to `:1234` or `:11434`: every stage is tested through
the real worker and the real datasets rules over a fake registry / GitHub
transport, a fake embedder and an idleness switch. **Not yet run live.**
Every "exists" below was read in the code on 2026-09-27.

## 0. Status: what was built, and where it departs from the design

Operator, 2026-09-27: "All of your recommendations above are sound, build it
with an agent." Section 11's decisions were answered the same day (listed
there); the code follows them.

| id | piece | where | tests |
|---|---|---|---|
| A | kind `package`: its stage path, its questions (locator + licence only), `create()` enqueues `package.resolve`, the JOIN blockers, the sweep | `mcp/datasets.py` (`PACKAGE_STAGES`, `PACKAGE_ENQUEUE`, `stages_of`, `package_notes`), `mcp/onboarding.py` (`blockers`, `after_job`, `sweep`, `tick`) | `mcp/test_onboarding.py` |
| B | `jobs.not_before` + `jobs.defer` (no attempt spent) + `jobs.claimable`; `mcp/idle.py`; `worker.run_one`'s idle check and `worker.Deferred`; `skills.enqueue`'s idle flag; the skill document index rebuild idle-gated | `mcp/jobs.py`, `mcp/idle.py`, `mcp/worker.py`, `mcp/skills.py`, `mcp/skill_match.py` | `mcp/test_onboarding.py` [queue], [idle], [rate] |
| C | resolve: npm / PyPI / GitHub links, pins, the version and commit rules, siblings | `mcp/package_resolve.py`, `onboarding.handle_resolve` | [resolve] |
| D | the licence from verbatim quotes (the LICENSE file and the manifest line at the commit, the tarball's LICENSE); `deps.fetch_verified` (dist.integrity, dist.unpackedSize, LICENSE extraction) | `package_resolve.licence_quotes`, `mcp/deps.py` | [resolve], [index] |
| E | the index stage (idempotent skip) | `onboarding.handle_index` | [index] |
| F | `packages.json` + the five hard-coded maps migrated as its SEED; the taxonomy from data; `tag/6+<sha8>`; `held.json`; `vocabulary.json`, the diff, the promotion floor; stat-based reload | `mcp/package_registry.py`, `skill_packages`, `skill_classify`, `skill_match`, `skill_select`, `skill_prompts`, `bench/skills/eval_packages.py` | `mcp/test_package_registry.py`; through the stages: `mcp/test_onboarding_registry.py` |
| G | sources: the GitHub tree at the commit (truncation walked), tiers 1-3, the README lead, the tarball's SKILL.md without a tag, `create_child` / `new_child_version` meta copy, `skills.repoint`, the deferred archive (`to_retire`), the retire stage, the name-major filter reading `package_version` | `mcp/package_sources.py`, `mcp/skills.py`, `mcp/skill_pipeline.py`, `onboarding.handle_sources` / `handle_retire` | [sources], [join], [replace], [decompose] |
| H, I | examples, the kNN label index, k by leave-one-group-out | `mcp/package_examples.py`, `mcp/example_knn.py` | `mcp/test_example_knn.py` |
| J | the vote reaches `plan_turn` (a weak area, a rank tier) + `x_yamadori.skills.knn` | `mcp/skill_match.py`, `mcp/skill_select.py` | `mcp/test_knn_selection.py` |
| K | the evaluation by leave-one-group-out | `mcp/package_eval.py` | `mcp/test_package_eval.py` |
| L | the dashboard API | `mcp/dash_skills.py` | [api] |
| M | the Skills page: PROMPT + LINKS, onboarding cards, `/skills/onboarding/<id>` | `web/src` | vitest, `mcp/test_dash_static.py` |
| O | this document, AGENTS.md, SKILL-FACTORY.md; an OPT-IN live check | `mcp/test_live_stack.py --only onboarding --onboard "<prompt>"` | not run |

DEPARTURES FROM THE DESIGN (each an operator decision or a fact the build
found):

- **No held-out split** (decision 1): leave-one-GROUP-out cross-validation
  over the example groups replaces 6.3's fixed split; there is no
  `split.json` and no fraction. The served kNN index uses every group, its k
  chosen by leave-one-group-out over them; the evaluation holds each group
  out once and chooses that fold's k by an inner leave-one-group-out over
  the other groups only (nested). 6.3, 6.4 and 7 below read that way.
- **Aliases bind to a package.** An alias typed `alias=package` belongs to
  that package; an unbound alias belongs to the package only when the prompt
  names ONE (`package_resolve.aliases_for`): with several, which package a
  bare word means is written nowhere, and it is not guessed.
- **The seed stays in code.** `package_registry.SEED` holds the hand rows;
  `packages.json` ADDS to them (a seed package's term and area never
  change), so a missing file changes nothing and every existing suite runs
  byte-identically without it.
- **`IDLE_MINUTES`** (decision 6): `mcp/idle.py` reads
  `skill_learn.IDLE_MINUTES` (15) and names its source: operator,
  2026-09-27, "keep ours (no external evidence)" -- the bonsai-ada-surgery
  repository, read first at his request, has no idle scheduler.
- **A restricted licence holds clarify for a person**, as the dataset assist
  does (`onboarding.held_for_person`, `worker.HOLD_RESTRICTED`); the design
  only said it is surfaced.
- **The old version leaves the held set at `retire`**, through the same
  promotion floor (a REPLACE keeps serving the old vocabulary until then).
- **The `parent` column** is not used: no stage job spawns a job of the
  dataset's own (a sibling onboarding is its own dataset, in the same
  `group`; a skill's jobs carry `skill:<id>`). The evaluation's model-decider
  arm is a job of the dataset at stage `evaluate`
  (`package.evaluate_decider`, gpu, idle-gated), so the onboarding completes
  only once it ran.
- **The kNN query unit is a FILE** (`example_knn`): every kept example file
  is embedded once as a query (import lines stripped, its first 4,000
  characters -- `skill_match.rank_all`'s state bound -- with the query
  instruction) beside the plain chunk documents, so the k chosen at build,
  the evaluation's nested leave-one-group-out and the served `vote()` all
  query the same shape: a piece of code. A group's vote is its best chunk.
  `vote()` strips import lines from its query too (the indexed documents
  have none; imports are detect()'s job).
- **A group is its repository + group path, not the commit**: the same demo
  at two versions is one group, so a held-out group never votes for its own
  earlier copy.
- **Linked example directories are added to the naming convention**, not
  used instead of it (both recorded).
- **The significance level** of the paired test is `package_eval.ALPHA` =
  0.05, the conventional two-sided level the repo's reported p-values are
  read against: operator, 2026-09-27, "keep ours (no external evidence)"
  (`ALPHA_SOURCE`). With it an exact McNemar test needs 6
  discordant pairs, so a package with fewer items says "too small to support
  a claim" and reports counts.
- **The vote reaches the selector only as a weak area and a rank tier**
  (`skill_match.plan_turn(knn=...)`, `skill_select.knn_vote`): a kNN-opened
  area pushes no lead and does not spend the package's first appearance, so
  with the stub decider live delivery is unchanged; `x_yamadori.skills.knn`
  records what the vote proposed. It queries the index only on a turn whose
  newest evidence holds JS/TS code (a fence, a tool call's code or file
  read, an import, a path), and only where the embedding stage is on.

Operator, 2026-09-27: "all of this should also happen when I add a new skill
prompt on the skills dashboard. We should be able to automate all of this
from a prompt with some links." And: "That is a stateful pipeline in action,
we already have the durable machine too." So onboarding is **one more staged
pipeline on the existing machine** -- a dataset row of a new kind whose stages
are jobs on `mcp/jobs.py`'s queue, advanced by `mcp/worker.py` exactly as a
dataset or a skill version is advanced today. There is no new orchestrator.

"All of this" is what was done by hand for koota, pmndrs/math, R3F and TSL
(`skills/ingested/pmndrs/spec.json`, `mcp/skill_packages.py`,
`bench/skills/eval_packages.py`):

1. resolve the prompt's links to `package@version` (npm, PyPI, a GitHub tag
   or commit), with a verbatim licence;
2. index the package source (`deps.py index --embed`, the registry history for
   the unseen-package rule);
3. refresh the package detector's vocabulary (unique exported symbols, the
   import rule);
4. discover the skill sources (SKILL.md and its references, llms.txt, docs at
   the tag) and run them through the ONE skills pipeline, retiring older
   versions only after their replacements arm;
5. fetch the package's EXAMPLES at the pinned version, label each by its OWN
   imports, and split them into a kNN label index and a held-out set;
6. evaluate on the held-out set, with results on the dashboard;
7. rebuild the skill document index and the kNN index (gpu lane, idle-gated).

---

## 1. The shape: a dataset of kind `package`

`mcp/datasets.py` already is "the corpus pipeline as data: what a dataset is,
and where each one has got to": a row with the operator's `prompt`, a
`source_url`, a `licence` that is a BLOCKER when unknown, a `stage`, `counts`
only a worker writes, and the rule that **a stage is left only when its job is
`done`** (`blockers()`), that **`errored` is not `done`**, and that re-running
an errored job is an explicit act (`rerun()`). `worker.advance_after()` moves
a dataset on the moment its stage job finishes. The dashboard already reads
it (`/dash/api/datasets/<id>`: jobs by state, job rows with `result` and
`error`, missing fields, warnings, next stage).

Onboarding is a dataset of `kind = "package"`, one row per resolved
`package@version`. Its stages:

```
submitted -> resolve -> clarify -> index -> vocab -> examples -> knn
   -> sources -> skills -> retire -> rebuild -> evaluate -> complete
```

| stage | lane | job queue | kind of work | what it waits on |
|---|---|---|---|---|
| resolve | net | `package.resolve` | deterministic | its job |
| clarify | -- | (none) | HUMAN only when no licence quote was found | `missing()` for kind package: the licence |
| index | gpu, idle | `package.index` | deterministic (embedder) | its job |
| vocab | cpu | `package.vocab` | deterministic | its job |
| examples | net | `package.examples` | deterministic | its job |
| knn | gpu, idle | `package.knn` | deterministic (embedder) | its job |
| sources | net | `package.sources` | deterministic discovery; creates skill rows | its job |
| skills | -- | (none: a JOIN) | the skills pipeline's model stages, per skill | every child skill terminal |
| retire | cpu | `package.retire` | deterministic | its job |
| rebuild | -- | (none: a JOIN) | the existing index schedules do the work | both indexes fresh |
| evaluate | cpu (+ gpu, idle, for a model-decider arm) | `package.evaluate` | deterministic (stub decider) | its job |

Why this order: the vocabulary must be live before the skills' activation
tests run (a LEAD skill's should-cases are checked through
`skill_packages.detect`, `skill_pipeline._lead_activation`), the taxonomy term
must exist before the tag stage files a skill under it, and the evaluation
must run against the store and indexes as they will be served. Examples and
the kNN build need no skill (a chunk's skill labels are computed at load from
the armed set, section 6), so they run before the skill model stages and all
gpu work stays serialised on the one gpu lane.

### 1.1 Which existing primitive each stage uses

| stage | existing primitives (read 2026-09-27) | addition |
|---|---|---|
| (entry) | `datasets.create(prompt, kind=...)`; `dash_skills._submit` | `create()` branches on kind `package`: no `dataset.fetch`, no `dataset.assist`; enqueues `package.resolve` with `dataset=did, stage="resolve"` |
| resolve | `jobs.add`; `deps.registry_history` / `history_of_packument` (packument reading); `skill_packages._PIN_AT` (`name@1.2.3` in prose); `skills.normalise_url` (GitHub page -> raw) | a resolver for npm / PyPI / GitHub links (2.1); writes `resolution.json`; a prompt naming N packages creates N-1 sibling datasets (same prompt, a shared `group` in `notes`) via `datasets.create` and advances each past resolve |
| clarify | `datasets.missing()`, `answer()` (operator provenance), `warnings()` (RESTRICTED licences surfaced), `FIELDS`' "unknown is a BLOCKER"; `skill_pipeline.licence_of` (a verbatim `license:` line or SPDX line) | per-kind required fields (kind package requires `licence` and `source_url` only); the licence is filled as `evidence` by resolve, so clarify is passed with no human when a quote exists |
| index | `deps.index_package(name, version, embed=True)` (fetch, `code_files`, `index_code.py`, `index_health`, atomic install, `record_published`, `record_history`); `gpu_room.use` (via `code_search`) | `deps.fetch`: verify the tarball against the packument's `dist.integrity`, extract `LICENSE*` / `COPYING*`, refuse an unpacked size above the packument's own `dist.unpackedSize` (2.4); skip when `is_indexed` and healthy and embedded (idempotent) |
| vocab | `skill_packages.vocabulary(rebuild=True)`, `held()`, `mapping_report()`; `bench/skills/eval_packages.py --labels` | the package registry file and the held manifest (section 5); a vocabulary FILE with a signature the proxy loads; promotion only if the standing labels do not regress (5.3) |
| examples | `skill_pipeline.fetch_source` rules (GET, robots, type, NUL refusal); `deps.is_minified`, `deps._dedupe`, `deps.BUILT_DIRS`, `deps.MAX_FILE_BYTES`; `skill_packages.imported_names`, `package_of_specifier`; `skill_screen` (credentials, hidden unicode) | discovery at the commit, labelling, the split (section 6) |
| knn | `index_code.chunk_file` (the package indexes' chunker, `MAX_CHUNK_LINES` sized to the embedder's window); `code_search.embed(is_query=False)`; `skill_match.doc_index`'s npz + signature pattern | `mcp/example_knn.py`: build, load, query, leave-one-group-out k (6.4) |
| sources | `skills.create(url=..., frontier=..., goal=..., meta=...)` (frontier path = decompose with a LEAD; `meta.package` is the declared lead package, `skill_pipeline._choose_lead`); `skills.new_version` + `store_source` + `enqueue` (the watch path's "new version while the served one serves"); `worker.licence_candidates` | GitHub tree listing at a commit (the SKILL-FACTORY follow-up "remote repo listing"); `skills.repoint(sid, url)` for a new package version (4.3); `create_child` copies `meta.onboarding` and `meta.package_version` beside `goal` and `package` |
| skills | the whole pipeline, unchanged: screen, screen_model, licence, decompose/distil, classify (tag/6, code-shaped topics), tests (tests/3, gates-aware), validate (mandatory faithfulness, quote repair, boundaries recorded), arm (supersedes) | the JOIN: `datasets.blockers()` for kind package at `skills` lists child skills still in `pipeline`, and child jobs that are `errored` (with the skill's own remedy); a worker sweep advances a package dataset whose blockers emptied (like `skill_pipeline.schedule_watches`) |
| retire | `skills.archive(sid, reason=, author=)` | `handle_decompose` records a vanished section's child as `to_retire` instead of archiving it at once when the source carries `meta.onboarding`; this stage archives them (4.3) |
| rebuild | `skill_match.index_state` / `schedule` / `handle_build` (gpu lane, run by the worker's minute loop); `skill_select.refresh_triggers` (already called by `handle_arm`) | `example_knn.index_state` / `schedule` / `handle_build`, registered like `skill_match` in `worker._register_*`; the idle gate on both builds (section 3) |
| evaluate | `bench/skills/eval_packages.py` (`--labels`, `--stub`), `bench/skills/replay_selection.py` (a copy of the live store, deterministic stages), `skill_packages.detect`, `skill_match.plan_turn` | `mcp/package_eval.py` (section 7), results JSON, `datasets.record_counts` for the summary |

The jobs table's `parent` column exists and nothing uses it today. Onboarding
uses it for the jobs a stage job spawns on behalf of the dataset (the idle
gate's deferrals keep the same row; see 3.2), so the dashboard can draw a
stage's children under it. `jobs.listing(dataset=did)` already returns every
stage job of the dataset.

### 1.2 Idempotency and resume

Every stage is a function of recorded inputs and writes its outputs
atomically (`os.replace`, as `deps.index_package` and `skills._write_atomic`
do), so a worker killed mid-stage leaves either the old output or none, and
`jobs.reclaim()` re-runs the stage with its attempts intact.

| stage | inputs recorded | re-run behaviour |
|---|---|---|
| resolve | the prompt, the links, the registry answers | the FIRST resolution pins the exact version and commit; a re-run keeps them unless the operator asks to re-resolve (a newer `latest` must never silently move a running onboarding) |
| index | name, version, integrity | `deps.is_indexed` + `index_health` + `embedded == 1`: skip |
| vocab | the held manifest's signature | same signature: the same file, no-op |
| examples | commit, the listing's blob SHAs | a file whose blob SHA is already stored is not fetched again; the split is written once per package@version and never re-drawn (6.3) |
| knn | the signature of the chunk rows | fresh signature: skip (the `skill_match.index_state` pattern) |
| sources | the discovered source list | a source already created for this dataset (same normalised URL in `meta.onboarding` skills) is not created twice |
| skills | -- | the skills pipeline's own resume: a version records its stage; `skills.rerun(sid, stage)` |
| retire | the `to_retire` lists | archiving an archived skill is a no-op |
| evaluate | the held-out file's SHA, the vocabulary and kNN signatures, the armed set's signature, the git SHA of the code | a result is keyed by those; the same key is not recomputed |

### 1.3 Model-assisted vs deterministic

Only the skills pipeline's model stages call a model (screen_model,
distil/decompose, tag, tests, faithful, quote_repair), each as today: the
model PROPOSES, the code verifies (quotes in the source, taxonomy values,
code-shaped topics named by the skill and its source, regexes that compile,
activation tests). Resolution, licence, indexing, vocabulary, example
discovery, labelling, the split, the kNN and the evaluation (stub decider) are
deterministic. Nothing in onboarding asks a model to name a package, an alias,
a version, a licence or a label: a wrong one there would be confidently
wrong in every later request (SkillsBench 2602.12670v4: self-generated skill
packs carried "confidently wrong" content; research doc 2.1).

---

## 2. Resolve and licence (stages resolve, clarify)

### 2.1 Links -> package@version -> commit

The resolver reads the prompt's URLs (and `name@1.2.3` pins in its prose,
`skill_packages._PIN_AT`) and classifies each:

| link | resolves to |
|---|---|
| `npmjs.com/package/<name>[/v/<version>]` | npm `<name>` at that version |
| `pypi.org/project/<name>[/<version>]` | PyPI `<name>` at that version |
| `github.com/<o>/<r>` (root, `tree/<ref>`, `releases/tag/<tag>`, `commit/<sha>`) | the repo at that ref; the package name from its `package.json` / `pyproject.toml` at the commit (raw GET); if that name@version is on the registry, the REGISTRY artefact is indexed (it is what gets installed: `deps.fetch`'s docstring) |
| a `SKILL.md`, a `tree/` folder, a docs page | a skill source (section 4), attached to the package the other links resolve to; with no package in the prompt, it is an ordinary skill submission and the package stages are skipped (recorded, not errored) |

VERSION, in order: the version in the link or pin; a MAJOR the prompt states
("r3f v10": the newest version with that major, stable preferred when one
exists, else the newest prerelease -- today that is how
`@react-three/fiber@10.0.0-alpha.5` was chosen by hand); else the registry's
`dist-tags.latest`. Each resolution records which rule chose it.

COMMIT for "at the tag", in order: the packument's `versions[v].gitHead`
(the commit npm recorded at publish); else a tag lookup in the packument's
`repository` among the forms our own spec already met -- `v0.6.6` (koota) and
`math@0.1.0` (pmndrs/math), `skills/ingested/pmndrs/spec.json` -- plus the
bare version; a monorepo package's `repository.directory` is recorded. No
tag: the docs and examples stages fall back to what the TARBALL ships (its
README, any `skills/**/SKILL.md` it carries -- `deps.SOURCE_EXTS` keeps `.md`)
and record "no tag for this version".

PyPI and GitHub-only repos resolve and record the same fields, but INDEX,
VOCAB and EXAMPLES are JS/TS-only today (`deps.fetch` reads the npm registry;
`skill_packages.vocabulary` reads JS exports and `.d.ts`; the import rule is
the JS/TS grammar). Those stages record "not supported for python" and the
onboarding continues with the skills; building them is item P in section 10.

### 2.2 The licence, verbatim

The licence is established the way the skills pipeline establishes it, never
proposed: `skill_pipeline.licence_of` over the tarball's `LICENSE` file (after
2.4's extraction fix) and the manifest's own `"license": "MIT"` line, each
kept as a quote with where it came from; a GitHub-only repo reads `LICENSE`
at the commit (`worker.licence_candidates`). No quote: the dataset holds at
`clarify` with the existing remedy (the operator's statement through
`POST /dash/api/dataset/answer`, recorded with provenance `operator`).
`datasets.warnings()` surfaces an AGPL, non-commercial, no-derivatives or
all-rights-reserved licence exactly as for any dataset; the skills
pipeline's own licence stage still fails a no-derivatives source for
distillation. Examples in a SEPARATE repository need their own quote from
that repository; without one they are recorded "unlicensed: not indexed".

### 2.3 Fetching: GET only

| what | how |
|---|---|
| registry metadata | GET `registry.npmjs.org/<name>`, `pypi.org/pypi/<name>/json`: documented APIs, the fixed UA of `skill_pipeline` (`yamadori-skill-worker/1`), no cookies, no auth |
| GitHub refs and trees | GET `api.github.com/repos/<o>/<r>/git/...` (ref lookup, one recursive tree at the commit). Unauthenticated, the REST API allows 60 requests an hour per address (GitHub's REST docs; not re-checked in this pass); an onboarding makes a handful. An optional `YAMADORI_GITHUB_TOKEN` is sent ONLY to `api.github.com`, never logged, never stored in a result. A truncated recursive tree (GitHub truncates very large trees) is walked non-recursively below the example and docs directories only |
| files at a commit | GET `raw.githubusercontent.com/<o>/<r>/<sha>/<path>`, one file at a time, through `skill_pipeline.fetch_source`'s rules for documents and a per-file cap of `deps.MAX_FILE_BYTES` for code (the cap `index_code.iter_files` already applies) |
| docs pages | `skill_pipeline.fetch_source` (robots.txt obeyed, type and size checked) -- the skills pipeline's own fetch stage |

Nothing is POSTed, nothing is installed, no lifecycle script or example is
run. A commit SHA in every URL makes a fetched file immutable.

### 2.4 Integrity and size (additions to `deps.fetch`)

`deps.fetch` today downloads the tarball with no integrity check and no size
limit. Onboarding adds: verify the bytes against the packument's
`dist.integrity` (SHA-512, SRI form) and record it; refuse when the unpacked
bytes exceed the packument's own `dist.unpackedSize` when present (a
consistency check against the registry's record, not a number of ours);
extract `LICENSE*` and `COPYING*` beside `SOURCE_EXTS`. The existing path-
traversal and extension filters stay.

---

## 3. GPU stages wait for an idle stack

### 3.1 What exists

- The gpu lane runs one job at a time across every worker (`jobs.claim` counts
  running rows inside the write lock).
- A measurement run pauses the gpu lane (`jobs.pause`, expiring).
- `skill_learn.idle_state` already defines idle for its learner: no client
  request (corpus `events` kind `turn`) for `IDLE_MINUTES`, and no other gpu
  job queued or running. `IDLE_MINUTES` = 15 is flagged in
  `docs/CONSTANTS-AUDIT.md` as "A-flag ... 15 ours; operator to confirm";
  onboarding reuses the value and inherits the flag, and adds no number.
- `gpu_room.use` leases the A4000 (the resident embedder) and answers
  `A4000_BUSY` / `A4000_NO_ROOM`.

What does NOT exist: a way for a job to wait without burning an attempt.
`jobs.fail(retry=True)` counts an attempt, so a stage that found the stack busy
three times would land in `errored`.

### 3.2 Additions

1. `jobs`: a `not_before REAL` column (added the way `datasets` adds
   `assist`: `ALTER TABLE` when missing), `claim()` skips rows whose
   `not_before` is in the future, and `jobs.defer(job_id, until, why)` returns a
   running job to `queued` WITHOUT counting an attempt, with `progress =
   "waiting for an idle stack: <why>"`.
2. `mcp/idle.py`: `stack_idle(job_id) -> {idle, why, until}`, extracted from
   `skill_learn.idle_state` (which then calls it), adding one check:
   llama-server's `/slots` `is_processing` (the watchdog and `slots` already
   read it) so a long generation that started before the quiet window is not
   overrun. `until` is DERIVED: the last request's time plus `IDLE_MINUTES`,
   or the next poll when a slot is generating.
3. `worker.run_one`: a job whose payload says `"idle": true` is checked BEFORE
   its handler; not idle -> `jobs.defer`. One place, so every gpu stage of
   onboarding -- `package.index`, `package.knn`, the evaluate stage's
   model-decider arm, and the skill model stages of an onboarding's skills
   (`skills.enqueue` sets `idle` when the skill carries `meta.onboarding`) --
   waits the same way. A job already running is never interrupted (the
   `jobs.pause` rule).
4. The two index rebuilds (`skill.match_index`, `package.example_knn_index`)
   are idle-gated too, per the operator's step 7. This CHANGES today's
   behaviour: after an arm, the skill document index stays stale until the
   stack is idle, and the request path runs as it does with a stale index
   (`rank_all` records why; the stub keeps the delivery). The operator
   decides (section 11).

Known risk: `deps.index_package --embed` is one subprocess that can run for
hours on a large package (its timeout is 14,400 s); once started it is not
idle-gated between batches. `index_code.py` has an `INDEX_APPEND` mode that a
batch-level yield could build on; not designed here.

---

## 4. Skills (stages sources, skills, retire)

### 4.1 Discovering the sources

At the resolved commit (or from the tarball when there is no tag), in tiers:

| tier | source | path through the pipeline |
|---|---|---|
| 1 | every `SKILL.md` in the repo (a `skills/<x>/SKILL.md` convention: koota, pmndrs/math) and the `references/*.md` beside it | the SKILL.md: frontier (decompose, with `meta.package` so its first part yields the LEAD); each reference: distil |
| 2 | every page the PROMPT links | the path its URL implies today (a SKILL.md or `tree/` -> frontier, else distil) |
| 3 | `llms.txt` at the repo root or the docs site named by the manifest's `homepage`: the pages it lists | distil, one skill per page |
| lead fallback | the README at the commit, when tier 1 found no SKILL.md | frontier with `meta.package` (decompose/4 writes ONE compact lead from the opening sections) |

Tier 3 is ingested ONLY when tier 1 found nothing: an author's own SKILL.md
is the curated set, and curated skills are what helps (SkillsBench: curated
+16.6 pp overall, self-generated -8.1 to -11.5; research doc 2.1), while every
added skill costs routing accuracy (Skill Shadowing 2605.24050: -0.21 pass
rate at 202 skills, from picking the wrong skill; Scaling Laws 2605.16508:
accuracy falls as a - b ln N). When tier 3 is skipped its page list is
recorded on the dataset and one click ingests it. The author's SKILL.md is
not trusted as written: 91.8% of 138K public SKILL.md files have at least one
defect (2608.08453 [A], research doc 2.1), so it walks the full pipeline.

Each source becomes `skills.create(url=<raw URL at the commit>, goal=<the
prompt's prose>, frontier=..., watch_hours=0, meta={"onboarding": did,
"package": name, "package_version": v})`. `watch_hours = 0` because a
commit-pinned URL cannot change; a new VERSION of the package is a new
onboarding (4.3).

### 4.2 What the pipeline already guarantees

Unchanged, and each is why a hand-held onboarding is not needed:
screen (deterministic, then the model screen; zero-width typography stripped,
tag and bidi characters quarantine), licence from a verbatim quote,
decompose/4 with ONE compact lead (`metadata.yamadori.lead_for`; SkillsBench:
compact +19.0 vs comprehensive +0.7), code-shaped topics named by both the
skill and its source (tag/6), tests/3 written against the skill's own gates,
faithfulness MANDATORY (switched off -> the version fails; research Part 4.5),
one quote-repair round, boundaries recorded (`validate.boundary`,
`validate.siblings`), arm on pass, quarantine on a failed screen or failed
activation tests. No review stage.

### 4.3 A new version of a package: replace or alongside, retire after arm

The machinery exists in the watch path: a changed source becomes a NEW VERSION
of the same skill, "the served one keeps serving until the new one arms"
(`skills.arm` supersedes), and a changed frontier source re-decomposes so that
a same-named child becomes a new version of itself
(`handle_decompose` -> `new_child_version`).

- REPLACE: the sources stage finds the earlier onboarding's source skill for
  the same package and repository path (`meta.package`, the path in its URL)
  and gives it a new version at the new commit (`skills.repoint` -- new, a
  small update of `source_url` -- then `new_version(origin="watch" |
  "watch_frontier")`, `store_source`, `enqueue`). Every child keeps serving
  its old version until its replacement arms. A child whose section is gone
  from the new source is today archived AT decompose time; with
  `meta.onboarding` it is recorded as `to_retire` and the retire stage
  archives it after the skills JOIN, i.e. after every replacement has armed
  or stopped.
- ALONGSIDE: new source skills; the old ones stay (R3F v9 and v10 are both
  held and both have skills today). Then `skill_select`'s structural filter
  "a skill whose NAME carries a major other than the one asked is skipped"
  should read `package_version` from metadata instead of the name (small
  change), because onboarded skill names will not always carry the major;
  siblings across majors are the "right family, wrong skill" risk
  (2606.10388: 95.0-95.7% of helpful top-three hits also carry the risky
  sibling).

Which one applies is an OPERATOR DECISION (section 11). Proposal: the form
has a `replaces` choice defaulting to REPLACE when the held version has the
same major and ALONGSIDE when it does not.

---

## 5. The detector's vocabulary and the package registry (stage vocab)

### 5.1 What is hard-coded today

Adding koota and math took code edits in five places, and a sixth package
would too:

| where | what |
|---|---|
| `skill_classify.VOCAB` | a `Term("koota", ..., "framework", regex, packages=...)` per package; the taxonomy is "fixed" |
| `skill_packages.TERM_PACKAGE` | term -> npm name |
| `skill_packages.CANONICAL` | package -> its skill (only when no `lead_for` skill exists) |
| `skill_match.PACKAGE_AREA` / `AREA_PACKAGE` | package <-> selector area |
| `bench/skills/eval_packages.py` `PACKAGE_AREA` | the same map, copied |
| `skill_prompts.TAG_VERSION` | bumped when the taxonomy changed (tag/3: "koota and pmndrs_math") |

### 5.2 One data file

`index/packages/packages.json`, written only by the vocab stage (and by hand
for the packages already held, migrated once from the tables above, which
then read it):

```json
{"<npm name>": {"term": "<slug>", "label": "<display name>",
                "aliases": ["<what the operator typed>"], "area": "<slug>",
                "versions": ["<held versions>"],
                "built_on": ["<held packages in its peerDependencies>"],
                "built_on_from": "package.json peerDependencies at <commit>",
                "onboarding": "<dataset id>",
                "licence": {"spdx": "<SPDX>", "quote": "<verbatim>", "where": "<file>"}}}
```

- The taxonomy term id is the package's slug; its WORD regex is built from the
  exact npm name and the ALIASES THE OPERATOR TYPED in the form (never
  proposed by a model: an alias is a detector rule). A one-word npm name that
  is an English word gets the `pmndrs_math` treatment automatically: only
  forms that cannot mean anything else (`name@<digit>`, a subpath import,
  `npm install name`), as `skill_classify` already does by hand for `math`.
- `built_on` comes from the package's own manifest (`peerDependencies`),
  cited, replacing a hand-written `BUILT_ON` row for new packages.
- `skill_classify.VOCAB`'s framework terms for packages, `taxonomy()`,
  `TERM_PACKAGE`, both `PACKAGE_AREA` maps and
  `CANONICAL` read the file; the hand rows become its first entries.
- `skill_prompts.tag_system()` already renders the taxonomy from
  `taxonomy()`; the pinned template text becomes the template with the
  catalogue as a placeholder, and the recorded version becomes `tag/6+<sha8
  of the catalogue>`, so a classify record says exactly which catalogue filed
  it and adding a term needs no hand-bumped version.
- No IMPLIES row is ever added automatically (`docs/CONSTANTS-AUDIT.md`
  removed every row not stated by the operator).

### 5.3 The vocabulary file and its promotion

`skill_packages.vocabulary()` builds once per process from every directory
under `index/packages/_src` (`held()` globs it). So a fetched package would be
"held" by the next process that starts, whatever onboarding decided. Changes:

- `held()` reads `index/packages/held.json` (the promoted package@version
  list) and falls back to the glob only when the file is absent.
- The vocab stage builds the CANDIDATE vocabulary with the new package and
  writes `vocabulary.next.json` with its signature and a DIFF: the new
  package's unique, code-only and dropped counts (`per_package`), and the
  names EXISTING packages lose -- uniqueness is relative, so a new package can
  make `useQuery` shared and blind the detector to koota.
- PROMOTION is a test floor, like a Laya head that is promoted only if it
  passes `bench/test_laya_head.py`'s floors, and like activation tests that
  quarantine a skill -- not a review: the candidate is promoted (written to
  `held.json` + `vocabulary.json`) unless `eval_packages.py --labels` on the
  standing labels (`bench/skills/package_detect_labels.jsonl`) loses a true
  positive or gains a false positive. Held, the dataset's blocker names the
  rows and the names lost; the operator may force it.
- The proxy loads `vocabulary.json` when its signature changes (a stat per
  minute), never builds it on the request path.

---

## 6. Examples and the kNN label index (stages examples, knn)

Operator: "r3f's examples are mostly drei; TypeGPU, Koota and drei have many"
-- label each by its OWN imports (multi-package rows); split into an
embedding kNN label index, RouteLLM-style (the nearest examples vote
packages/skills into the selector's candidates; human-written code only; no
model-written queries), and a held-out evaluation set.

### 6.1 Discovery at the pinned commit

- The directories the PROMPT links come first. Otherwise, in the recursive
  tree at the commit, directories whose name marks examples (`example`,
  `examples`, `demo`, `demos`, `sandbox`, `sandboxes`, `stories`, a docs
  app's `examples`). This list is a starting CONVENTION, to be checked against
  the repositories of the packages we hold when it is built; the tree listing
  and the chosen directories are recorded, so a miss is visible.
- Files: `.ts .tsx .js .jsx .mjs` (the JS/TS family `deps.imported` reads).
- HUMAN-WRITTEN ONLY: out go `deps.BUILT_DIRS`, `node_modules`, lockfiles,
  anything `deps.is_minified` flags, files headed "generated" /
  "@generated" / "auto-generated", and near-duplicates by `deps._dedupe`'s
  rule. Nothing a model wrote enters (Tool-DE 2510.22670: LLM-written example
  usage "provides the smallest (often negative) gains"; Doc2Query--: filtering
  hallucinated queries gives up to 16%; research doc 2.2).
- SCREENED AS DATA: `skill_screen`'s credential rule (a demo API key drops the
  file), `strip_typography`, and a tag or bidi character drops the file.
  Example text never reaches a model's prompt; it is embedded and stored.

### 6.2 Labels from the example's own imports

- The EXAMPLE GROUP is the split unit: the directory directly under an
  examples root for a multi-file demo, else the file. Every chunk of a file
  and every file of a group share one side of the split.
- A file's imports are read with `skill_packages.imported_names` (the binding
  -> specifier map, namespace imports included) and
  `package_of_specifier` (subpaths: `three/tsl`, `three/webgpu` -> TSL).
- A CHUNK's package labels are the HELD packages whose imported bindings the
  chunk references (`useFrame` -> fiber, `OrbitControls` -> drei, `THREE.x` ->
  three). One chunk may carry several: multi-package rows. A chunk that
  references no imported binding carries no package evidence and is not
  indexed.
- Imports of packages we do NOT hold are recorded per example: they are the
  "onboard next" list (7.3).
- The IMPORT AND REQUIRE LINES ARE REMOVED from the indexed text: they are the
  label's source, and a vector that saw them would learn the label from its
  own answer. The evaluation strips them the same way (7.1).
- SKILL labels are not stored. The stage stores each chunk's code-shaped
  identifiers; at load, a chunk's skills are the ARMED skills of its packages
  whose verified topics are among them (`skill_classify.code_shaped` topics,
  the same rule the tag stage verifies with). A skill that arms later labels
  old chunks with no rebuild.

### 6.3 The split

**SUPERSEDED (operator decision 1, 2026-09-27): there is no split.** Every
example group is held out once, in leave-one-GROUP-out cross-validation,
and its votes come from the other groups only; k is chosen inside the
training folds. The paragraph below is the design as first proposed.

Deterministic by group: a hash of `(repository, commit, group path)` assigns
each group to TRAIN (the kNN index) or HELD-OUT (the evaluation). The split is
written once per `package@version` with its SHA-256, and never re-drawn after
any result exists. A group labelled with several packages is held out for all
of them. The FRACTION is an operator decision (section 11): no measurement in
this repo sizes it. What sizes it is PROTOCOL rule 4 -- the held-out n must
be large enough to detect the difference the evaluation claims -- so the
evaluation prints each package's held-out n and the smallest paired
difference it could detect, and a package too small to support a claim says
so rather than reporting a rate.

### 6.4 The index

- Chunks: `index_code.chunk_file` (the chunker every package index uses;
  `MAX_CHUNK_LINES` sized to the embedder's window).
- Documents embedded PLAIN, the query with an instruction (the Qwen3-Embedding
  card: an instruction on the query side only, 1-5%; research doc 2.2): task
  `"Given code an agent is reading or writing, retrieve example code that
  uses the same libraries"` -- a versioned text (`knn/1`), unmeasured wording.
- Stored as `index/skills/example_knn.npz` (vectors, chunk ids, signature) and
  `example_knn.rows.jsonl` (chunk id, group hash, package labels, code-shaped
  identifiers, repository path at the commit) -- the `skill_match_docs.npz`
  pattern, rebuilt by an idle-gated gpu-lane job when stale.
- `k` is MEASURED, not chosen: leave-one-GROUP-out on the TRAIN partition
  (each train group scored against the others, its own group excluded), over
  every k from 1 to the largest group count of any package, maximising the
  vote's top-1 package accuracy; the chosen k is stored with its n and the
  curve. The held-out partition is never read to choose it.

### 6.5 How votes reach the selector, with no threshold

`skill_match.plan_turn` already has the two slots a vote needs: package areas
(HARD when the package is named, imported, pinned or in an error; WEAK "for
the decider to confirm", with "None of these" offered) and a rank order inside
each question (a package's lead first, then cosine, then the pattern path's
order).

- WHEN: only on a turn whose newest evidence (`skill_select.evidence_view`, a
  dependency's own files set aside) holds JS/TS code the model read or wrote
  -- the examples' own language. The rule `open_areas` applies to every area:
  code shows it, or it is not open. A prose turn does not query the index.
- THE VOTE: the query is that code (the same 4,000-character state bound
  `skill_match.rank_all` applies); the top-k chunks, one per GROUP (its best
  chunk), each vote for their packages; the tally is ordered by votes, ties by
  best rank. Rank only, no cosine cut (ToolRet, SkillRet and the calibration
  papers: a score depends on the query and the batch; research doc Part 3).
- INTO THE CANDIDATES:
  1. the package at the TOP of the tally (argmax; all of them on a tie) that
     `detect` did not already put in play opens as a WEAK area, `why: "knn:
     m of k neighbours"`;
  2. inside any package question, skills that the neighbours' chunks label
     rank after the lead and before the cosine order.
  Nothing is delivered on a vote alone: a weak area is a soft question. With
  the STUB decider (the default), a weak area delivers nothing -- the stub
  answers only with the pattern path's picks -- so live delivery is
  unchanged until the Bonsai decider replaces the stub, and the record shows
  what the kNN proposed: `x_yamadori.skills.knn {index, k, groups: [{rank,
  packages}], tally, opened_weak}` (group hashes, never example text).
- WHY a non-parametric vote and not a trained router: RouteLLM (2406.18665v4,
  research doc 2.5) found "high capacity approaches performing worse in a
  low-data regime" (BERT and an 8B causal LM near random on its Arena data);
  a package's examples are that regime. Tool2Vec (2409.02141) represents a
  tool by its usage embeddings (Recall@3 91.84 vs 79.97). That kNN over
  example code transfers to our turns is INFERENCE, which the evaluation
  measures (7.1).

---

## 7. Evaluation on the held-out set (stage evaluate)

Deterministic, offline, on a copy of the live store (`replay_selection`'s
harness). Results go to `index/packages/onboarding/<did>/eval/<key>.json` and
the summary to `datasets.record_counts`. The key is the held-out file's SHA,
the vocabulary, kNN and armed-set signatures and the git SHA.

### 7.1 What is measured

| measure | how | truth |
|---|---|---|
| detection, imports STRIPPED | each held-out file as an agent step (a tool call writing it), import/require lines removed; `skill_packages.detect` | the file's own import labels; precision / recall per package and overall |
| detection, imports KEPT | the same with imports | a sanity row: the import rule must give recall 1.0; less is a bug, not a result |
| kNN | held-out chunk -> the top-k TRAIN groups -> the tally | top-1 and recall of the true packages among the voted ones |
| detection + kNN | a package counts as found by detect OR by the tally's top | PAIRED with detection alone on the same items (PROTOCOL rule 6), the paired test, n printed |
| selection | each held-out file through `skill_select.decide` / `plan_turn` (stub decider) | package area opened; the lead given on the package's first appearance; a delivered skill among the file's derived skill labels; delivered skills of packages the file does not use (precision) |
| selection, model decider | the same with `YAMADORI_SKILL_DECIDER` set, as a gpu job (idle-gated), n >= 2 runs reported side by side (PROTOCOL rule 10) | as above; only when a decider other than the stub exists |
| regressions | `eval_packages.py --labels` and the daily eval (`bench/skills/test_daily_eval.py`) on the store with the new package | no MUST / MUST-NOT miss that the store before it did not have |

The derived skill label (a skill of the file's package whose topics appear in
the file) is a PROXY for "this skill would have helped", and the dashboard
says so. The example eval measures CODE evidence only; prose turns stay
measured by the daily eval and the package labels.

### 7.2 Keeping it honest

- SPLIT BY GROUP, fixed before any result (6.3). Chunks of one file, and files
  of one demo, never straddle it.
- NEVER TUNED ON IT: nothing that builds or chooses reads the held-out file --
  the vocabulary is built from the package SOURCE, the skills from its DOCS,
  `k` by leave-one-group-out on TRAIN. Any change made after a held-out result
  was read (a retag, a vocabulary rule, a skill edit) marks every later
  result on that set `in_sample: true` with the change named, exactly as this
  repo labels in-sample numbers (the route and daily evals); a fresh held-out
  claim needs a new package version's examples.
- LEAKAGE THROUGH THE DOCS: a docs page can embed an example verbatim. A
  held-out file with a run of lines found in a skill's source (the quote
  check's own normalisation) is reported in a separate column, "seen by a
  skill source".
- NO MODEL-WRITTEN QUERIES: the queries are the held-out human-written files.
- The numbers print their n, and a package whose held-out n cannot support a
  difference says so (PROTOCOL rule 4).

### 7.3 Coverage gaps

| gap | how it is found |
|---|---|
| helpers with no skill | the package's exported names the examples use (imported bindings and member calls in the vocabulary), across both partitions, that no armed skill names in its topics, items or body; ranked by the number of groups using them |
| detector blind spots | names the examples use that the vocabulary dropped as shared, platform or English (`per_package` dropped counts, now per name) |
| packages to onboard next | held-out and train imports of packages we do not hold, by group count |
| skill with no example | armed skills of the package whose topics no example uses (a candidate for a boundary or for retirement, never retired automatically) |
| black-hole candidates | `python mcp/skill_boundaries.py black-holes` restricted to the new skills |

---

## 8. The dashboard: API and flow

No review gate (operator, 2026-09-24; memory "no review gate"): everything
runs on submit; review is optional and after the fact, and the corrections are
the existing skill actions (edit, disable, archive, quarantine, re-run a
stage) plus re-running an errored stage job.

### 8.1 API (all under `/dash/api`, behind `accounts.identify`)

| call | body / answer |
|---|---|
| `POST /dash/api/skill` | EXISTING. New shape: `{"prompt": "Add koota 0.6.6 ... https://github.com/pmndrs/koota", "links": [...]?, "aliases": [...]?, "replaces": "replace" \| "alongside"?}` -> `{"ok": true, "onboarding": {id, stage, packages: []}}`. The URL / text / frontier shapes are unchanged. A prompt whose links resolve to no package runs its skill sources as today's skill submissions (the resolve stage records it) |
| `GET /dash/api/skill-factory/onboarding` | every package dataset: `{id, group, package, version, stage, state, updated, counts}` |
| `GET /dash/api/skill-factory/onboarding/<id>` | the dataset detail `/dash/api/datasets/<id>` already returns (jobs by state, job rows with `result` / `error` / `progress`, blockers, warnings, next stage) PLUS `resolution` (links -> package@version, commit, the rule that chose each), `licence` (quote, where), `index` (health), `vocab` (the diff, promoted or held and why), `examples` (groups, files, chunks, per-package counts, train / held-out n, dropped with reasons), `knn` (k, its curve and n, index signature), `skills` (every child: id, name, status, version, lead_for), `retire` (archived with reasons), `eval` (7.1-7.3, each number with its n), `reviews` |
| `POST /dash/api/skill-factory/onboarding/review` | `{id, stage, note}`: an after-the-fact note, stored in the dataset's `assist.reviews`; changes nothing else |
| `POST /dash/api/dataset/answer` | EXISTING: the operator's licence statement |
| `POST /dash/api/dataset/rerun` | EXISTING: re-run one ERRORED stage job |
| `POST /dash/api/skill-factory/onboarding/promote` | `{id}`: force a HELD vocabulary (5.3), recorded with the author |
| `POST /dash/api/skill-factory/onboarding/tier3` | `{id}`: ingest the recorded llms.txt page list (4.1) |

No response carries a filesystem path (the `skills.public` rule); example
rows carry the repository path at the commit, which is public.

### 8.2 The Skills page

- CREATE gains a mode, `PROMPT + LINKS`, beside PASTE TEXT / URL(S) /
  FRONTIER: a textarea for the prompt, an optional aliases field (the words a
  person uses for the package; the detector's rule, so typed, not guessed), and
  the replace / alongside choice when a version of the package is held.
- WATCHING shows an ONBOARDING CARD per package (the `PipelineCard` idiom): a
  stage rail over the dataset's stages (`StageRail` drawn from the job rows);
  the waiting reason while a gpu stage waits ("waiting for an idle stack: the
  last client request was 3 min ago"); a blocker with its remedy and owner;
  the licence form when clarify holds; the child skills as they arm (links to
  `/skills/<id>`).
- `/skills/onboarding/<id>`: one panel per stage with its result -- the
  resolution table, the licence quote, the vocabulary diff (with the names
  existing packages lost), the example counts and the split, k and its curve,
  the skills, and the EVALUATION: detection P/R stripped and kept per package,
  the paired detection vs detection + kNN table with n and the test, the
  selection rows, the coverage-gap lists. A REVIEW note box per panel.

---

## 9. Data layout

| what | where |
|---|---|
| the onboarding row | `datasets` table (jobs DB), `kind = 'package'`; `prompt`, `source_url` (the first link), `licence`, `stage`, `counts`, `assist` (field provenance, reviews), `notes.group` |
| its stage jobs | `jobs` table, `dataset = <did>`, `stage = <stage>`, `parent` for spawned rows, `not_before` for deferrals |
| its skills | `skills` / `skill_versions`, `meta.onboarding = <did>`, `meta.package`, `meta.package_version`; folders under `index/skills/library/` as today |
| stage outputs | `index/packages/onboarding/<did>/`: `resolution.json`, `licence.json`, `vocab_diff.json`, `sources.json` (tiers, chosen and skipped), `examples/manifest.jsonl` (group, file, blob SHA, labels, unheld imports, split), `split.json` (+ SHA-256), `eval/<key>.json` |
| package index and source | EXISTING: `index/packages/<slug>@<ver>.sqlite3`, `index/packages/_src/<slug>@<ver>/`, `registry_history.json` |
| example files | `index/packages/_examples/<slug>@<ver>/<commit>/<path>` (text only) |
| registry of onboarded packages | `index/packages/packages.json` (5.2), `held.json`, `vocabulary.json`, `vocabulary.next.json` |
| the kNN index | `index/skills/example_knn.npz`, `example_knn.rows.jsonl` (beside `skill_match_docs.npz`) |

---

## 10. What exists and what must be built

EXISTS and is reused unchanged: the durable queue (lanes, claim, reclaim,
heartbeat, pause, errored semantics); datasets (stage, blockers, advance,
answer, rerun, record_counts, warnings) and its API; the worker (handler
registry, advance hook, minute scheduler); the whole skills pipeline
(fetch with robots, screen, licence from a quote, decompose with a lead,
distil, tag/6, tests/3, mandatory faithfulness, quote repair, boundaries,
arm/supersede, the watch path's new-version-while-serving); `deps`
(fetch, code selection, index with health, registry history); the detector
and its evaluation (`skill_packages`, `eval_packages.py`, the labels, the
replay harness, the daily eval); `skill_match`'s document index and its
upkeep job; `gpu_room`; the React Skills page's create / watching / stage-rail
components.

BUILD (agent-days, estimates, offline tests included; nothing live):

| id | piece | est. |
|---|---|---|
| A | `datasets`: per-kind stage paths and required fields, kind `package`, `create()` branch, the skills and rebuild JOIN blockers, a worker sweep that advances a package dataset whose blockers emptied | 1.0 |
| B | `jobs.not_before` + `defer()` + claim filter; `mcp/idle.py` (from `skill_learn.idle_state`, + `/slots`); `worker.run_one` idle check; `skills.enqueue` idle flag | 1.0 |
| C | resolve: npm / PyPI / GitHub links, version rules, gitHead / tag lookup, sibling datasets | 1.0 |
| D | licence from the tarball / manifest / repo; `deps.fetch` integrity, `unpackedSize`, LICENSE extraction | 0.5 |
| E | index stage wrapper (idempotent skip) | 0.25 |
| F | `packages.json` + migration of the five hard-coded maps; taxonomy from data; tag version with the catalogue's hash; `held.json`; vocabulary file, diff, promotion floor; proxy reload | 2.0 |
| G | sources: GitHub tree at a commit, tiers 1-3, README lead fallback, tarball SKILL.md, `create_child` meta copy, `skills.repoint`, deferred archive (`to_retire`), the retire stage, the name-major filter reading metadata | 2.0 |
| H | examples: discovery, filters, screen, groups, import labels, identifiers, split | 1.5 |
| I | `mcp/example_knn.py`: chunk, strip imports, embed, npz, leave-one-group-out k, schedule / build job, load | 1.25 |
| J | `plan_turn` integration (weak area on the tally's top, rank tier) + `x_yamadori.skills.knn` | 1.0 |
| K | `mcp/package_eval.py`: 7.1-7.3, keys, in-sample marking, leakage column | 2.0 |
| L | dashboard API (8.1) | 0.75 |
| M | web: the create mode, the onboarding card, the detail route, types and tests | 2.0 |
| N | tests: `mcp/test_onboarding.py` with a fake registry / GitHub transport and a tiny package fixture (the offline guard forbids network and the live state; stores at temp paths via `mcp/offline_stores.py`); the stage table end to end with deferrals, a killed worker, an errored child, a held vocabulary | 1.5 |
| O | docs: AGENTS.md, SKILL-FACTORY.md; a live check in `mcp/test_live_stack.py` that onboards one small package through the worker on an idle stack | 0.75 |
| | **total, JS/TS packages** | **~18.5** |
| P | Python packages: PyPI sdist/wheel fetch and index, a Python vocabulary (`__all__`, module-level defs via `ast`), the Python import rule, Python example discovery (deferred) | 3.0 |

Order of value: A, B, C, D, E, F, G, N make "a prompt with links" onboard a
package's skills with its detector (~9 days); H, I, K add the examples and
the honest evaluation; J lets the votes reach the selector; L, M, O finish
the surface.

---

## 11. Operator decisions this design needs

ANSWERED 2026-09-27 (operator: "All of your recommendations above are
sound, build it with an agent."):

| # | decision | built as |
|---|---|---|
| 1 | "Held-out split: no fraction is invented. Use leave-one-GROUP-out cross-validation over the example groups (every group is held out once, and its votes come from an index built without it). k for the vote is chosen inside the training folds only. If a fold count is ever needed, it's the number of groups." | `example_knn` (k by leave-one-group-out over all groups for the served index), `package_eval` (outer leave-one-group-out, inner k per fold) |
| 2 | "The same major version replaces the old one (the old version serves until the new one arms); a new major sits alongside (Right Family, Wrong Skill risk noted in the record)." | `package_resolve.replaces_rule`, `package_sources.make_skills`, `onboarding.handle_retire`; the risk is in `resolution.json` `replaces.risk` |
| 3 | "llms.txt-listed pages are pulled in only when the package ships no SKILL.md." | `package_sources.discover` (recorded otherwise; `tier3` ingests on a click) |
| 4 | "A new vocabulary is switched on only if the standing detection labels don't get worse (it can hold a vocabulary back, recorded with the reason)." | `package_registry.floor` / `promote`; `vocab.json` state `held` with `why`; the operator's force |
| 5 | "The skill doc-index rebuild is idle-gated (one GPU consumer at a time)." | `skill_match.schedule` enqueues `{"idle": true}` |
| 6 | "IDLE_MINUTES: don't take 15 as given. Use the existing idle definition (skill_learn.idle_state plus /slots) and mark the number with its source, or list it for the operator if there's none." | `mcp/idle.py`: `skill_learn.IDLE_MINUTES` + `/slots`; the 15 has no source beyond "15 ours" (docs/CONSTANTS-AUDIT.md), so it is LISTED FOR THE OPERATOR (`idle.IDLE_MINUTES_SOURCE`) |
| 7 | "The taxonomy grows automatically from package names plus the aliases the operator types." | `package_registry.add_package` / `words_rule`, `skill_classify.refresh` |
| 8 | "No GitHub token: unauthenticated GETs, with backoff on rate limits." | `package_net` (no Authorization header at all); `RateLimited` -> `jobs.defer` to the server's own reset time |
| 9 | "No size caps invented: record the bytes, bounded only by existing limits and the registry's own recorded size." | `deps.fetch_verified` (`dist.unpackedSize`, the per-member 2,000,000 of `fetch`), `deps.MAX_FILE_BYTES` per code file, `skill_pipeline.FETCH_MAX_BYTES` per document; bytes recorded in `index.json`, `examples.json` |

FOLLOW-UP DECISIONS (operator, 2026-09-27, after reading the
bonsai-ada-surgery repository, which has no idle scheduler, statistical
test or tool screen: "keep ours"):

- `IDLE_MINUTES` = 15: kept; its source is "operator, 2026-09-27: keep ours
  (no external evidence)" (`idle.IDLE_MINUTES_SOURCE`).
- `package_eval.ALPHA` = 0.05: kept, with the same source line.
- Code-tool results over a HELD PACKAGE are SCREENED (section 12): what the
  index tools return from a held package's index (`code_search.handle`
  with a `db` in the package store -- never the bound repository) passes
  `skill_screen.screen_fetched`; an offending line is removed ("[removed by
  the screen]", the tool output's code fences held aside so the cut is the
  line, not the whole file) and recorded in the result's `screen` field and
  `code_search.SCREENED`; a text the screen cannot cut clean is withheld as
  `QUARANTINED`. `mcp/test_package_screen.py`.

The questions as they were put (kept for the record):

No number below is ours to invent (memory "no invented numbers").

1. **The held-out fraction** of example groups (6.3). The evaluation will
   print, per package, the n it gets and the smallest paired difference that
   n can detect, so the first run shows what a given fraction buys.
2. **Replace or alongside** for a new version of a held package (4.3), and
   whether the proposed default (same major replaces, a new major goes
   alongside) is right.
3. **Tier-3 docs** (llms.txt pages) ingested automatically only when the
   package ships no SKILL.md (4.1), given the routing cost of every added
   skill.
4. **The vocabulary promotion floor** (5.3): a test gate that can hold a
   vocabulary, like activation tests and the Laya floors -- or promote always
   and let the daily eval in `run_tests.py` catch a regression afterwards.
5. **Idle-gating the skill document index rebuild** (3.2 item 4), which leaves
   newly armed skills out of retrieval until the stack is idle.
6. **`IDLE_MINUTES` = 15**, already flagged "operator to confirm" in
   `docs/CONSTANTS-AUDIT.md`; onboarding would lean on it for every gpu stage.
7. **The taxonomy grows from onboarding** (5.2): a framework term per package,
   its words from the npm name and the operator's typed aliases. Until now the
   taxonomy grew by hand.
8. **An optional GitHub token** (`YAMADORI_GITHUB_TOKEN`) for the tree and ref
   API.
9. **Archive and example caps**: none is invented here; the per-file cap is
   the existing `deps.MAX_FILE_BYTES`, the tarball is bounded by the
   registry's own `unpackedSize`, and every stage records bytes and counts so
   a cap, if wanted, can be set from what the first runs fetched.

## 12. Security, in one place

- Fetched content is DATA. Skill sources pass the skill screen (deterministic,
  then the model screen) before any model reads them, as today. Example code
  never reaches a prompt: it is embedded, stored and shown escaped; a file
  with a credential, a tag or bidi character is dropped. Package source is
  never executed (no install, no lifecycle scripts, no example run); the
  tarball extraction keeps its traversal and extension filters and gains an
  integrity check. A newly held package's source becomes readable by the
  second brain's code tools, as every held package's is -- and since
  2026-09-27 (operator) every held package's tool results pass
  `skill_screen.screen_fetched` (`code_search.screen_package_result`).
- GET only, fixed headers, robots.txt for pages, commit-pinned URLs; the only
  credential is the optional GitHub token, sent only to `api.github.com`.
- Licences from verbatim quotes only; an unknown licence holds the onboarding
  at clarify; restricted licences are surfaced by `datasets.warnings()`; a
  no-derivatives source fails distillation in the skills pipeline.
- Every write is recorded with its author (`accounts.identify`'s account id),
  as the skill factory already records submissions and edits.
- Nothing names a package, alias, version, licence or label by model: those
  are rules the detector and selector act on for every later request.
