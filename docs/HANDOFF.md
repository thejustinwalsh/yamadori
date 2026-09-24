# Handoff — 2026-09-22

The state of Yamadori at a compaction boundary. Written so the next run starts
from facts in files rather than from a summary of a summary.

## Run queue (2026-09-23, operator's order: top runs first)

Each run goes to completion. Nothing is stopped early; results are read at
the transcript level (docs/TRANSCRIPT-REVIEW-*.md) before the next change.

1. **Now:** bare arms, LiveBench (lb-20260923-minp0) and the domain suite
   (group overnight-0923-minp0), min_p 0.0.
2. **Next deploy, one proxy restart:** fan-out code-aware selection with the
   winner DELIVERED, plus the check_code tool and repair pass. Then the stack
   arms run to completion: LiveBench tier `max` (and `minimal`, `bonsai+check`),
   and domain A6 (plus a deep-thinking-only arm on TypeGPU, TSL and React).
3. **After the event, bug fixes only** (things that stop the machinery from
   working as designed): the proxy cancelling work on client disconnect, and
   the empty-answer root cause (task chips). **No behaviour change is made
   because of a result.** Router over-escalation (a common API name like
   `useFrame` sending a question to deep thinking, which then makes zero
   searches) is REPORTED with its evidence, not changed. The operator's rule:
   collect complete data in a working state first, show the evidence, decide
   later.
3b. **Sampling is owned by the proxy** (operator, 2026-09-23: "we don't let
   people change our defaults here because this is a fine-tune"). tiers.apply
   enforces the vendor's settings (PrismML Bonsai 2 card / Qwen thinking mode:
   temp 1.0, top_p 0.95, top_k 20, min_p 0; instruct values when thinking is
   off) on every request, client and internal alike. It overrides client
   values and reports them in x_yamadori.sampling. Fan-out variants no longer
   carry their own temperatures. The llama-swap `temperature?: 0.3` default
   is removed from config.yaml; that takes effect at the next llama-swap
   restart, though with the proxy always sending temperature it can't apply
   anyway.
4. **LAST, queued by the operator: KV-cache quantisation, max out the KV.**
   This reverses the earlier "never quant KV lower" decision, so it's an
   experiment with a gate, not a switch:
   - measure q8_0 (today, -c 163840) against a quantised KV at the largest
     context that keeps >= 1 GB free at peak on the 5060 Ti (the display is on
     the iGPU now), up to the native 262,144
   - bench/longctx sweep on both: tok/s and recall/reasoning accuracy vs
     context length
   - keep the larger context only where accuracy is not significantly below
     the 8k baseline; otherwise stay at the smaller one
5. **Then: bare-model SWE-bench with the GPU to itself, logged in full.**
   Verified Mini (50 instances, seed 20260923, results/mini50-* resume),
   arm bonsai, the leaderboard's mini-swe-agent config, trajectories and
   evaluation logs kept. About 100 min per instance under load; alone it
   should be faster, so measure it. At that rate the run takes days, and it
   needs the card to itself.

## 2026-09-22 evening — what changed since the sections below

Everything under this heading was checked against the code at about 19:30. The
sections after it are the earlier handoff and are left unedited as history.
Where they now say something false, the list at the end of this section names
it.

**Generation constraints were inventoried and replaced by one rule.**
`docs/CONSTRAINTS.md` is the evidence: every limit on the serving path, with a
verdict. The results are:

- `config.yaml` launches with `--reasoning-budget 8192` plus a budget
  message. It used to be 1000.
- `tiers.budget()` sends `max_tokens = R_CAP + max(client, A_MIN)`, with
  `R_CAP=8192` and `A_MIN=2048`, plus `reasoning_budget_tokens` and the
  message on every request.
- `mcp/model.py` is the one door for internal generation (`summarize_text`,
  shomen, the worker). It calls the same `tiers.apply()` the proxy uses and
  turns a `finish_reason: length` into a `BudgetEvent`.
- `reasoning_effort` values are parsed from the served template
  (`low`/`medium`/`xhigh`). `high` rounds up to `xhigh`, so the `high` tier
  no longer returns 500s.
- Every response carries `x_yamadori`: tier, effort sent, tool gate,
  hints, selection, fan-out, deep thinking, hops and budget.

**The selection engine exists.** The plan is `docs/SELECTION-BUILD.md`, and
the code is `mcp/selection.py`, wired in `proxy.prepare`. It works like this:

- The tier says what is allowed, and selection decides what fires.
- Deep thinking uses two signals: the regex plus a symbol lookup, and the
  trained Laya `route_in` head. When they disagree, deep thinking runs. When
  Laya is down, its signal is recorded as `None`.
- `delegate_investigation` is now OFF by default. Turn it on with
  `YAMADORI_DELEGATE_TOOL=1` or a tier's `delegate` flag, as a benchmark arm.
- The model's surface is the 8 MCP tools plus `record_step`, `read_rings` and
  `bind_project_context`. **Superseded 2026-09-24:** main gets only the
  client's tools and the image tools; our tools are the second brain's, and
  `bind_project_context` is deleted (AGENTS.md "One model, one cache").

**Laya `route_in` was retrained.** It now uses 289 labels. The staging copy
is in `index/laya_staging_20260922_191425`, and the old head is in
`index/laya/_backup_20260922_191425`. `bench/eval_route_heldout.py` scored it
on 120 held-out package-domain labels, investigate-vs-not:

| condition | score |
|---|---|
| new head | 80/120 |
| old head | 64/120 |
| `selection.decide` | 89/120 |

The head still does not beat the rule.

**The dataset pipeline runs.** The pieces are:

- `mcp/worker.py` exists and is supervised by the watchdog, which now
  watches five services.
- The `jobs.py` lane limit is global across processes.
- There is no review stage.
- `clarify` is model-assisted, and a licence is filled only from a verified
  verbatim quote.

**Packages are indexed.** Use `mcp/deps.py index name@version [--embed]` and
`mcp/deps.py health`. At 19:30, 17 package databases were on disk in
`index/packages/`, and `three@0.186.0` was being rebuilt.

**Other changes:**

- Concept seeds come from the 27B's own token embeddings
  (`scripts/extract_token_embd.py`, which writes `index/token_embd.npz`).
- The React dashboard is in `web/` and is served from `web/dist` at the site
  root (`/`, wouter routes; old `/dash/*` links 302 there). The Python pages
  are at `/dash/classic*`, the JSON API at `/dash/api/*`.
- Tests: `scripts/run_tests.py` runs every suite, and `--live` goes through
  `:1234` (`mcp/test_live_stack.py`).
- The watchdog is two-strike, and it will not restart llama-swap while the
  slot counters are moving.

**Tidy.** Superseded backups and stale root logs were moved into a gitignored
`attic/`, and `attic/README.txt` lists them. Nothing was deleted or
committed. The `/ui/` and "208k" comments were corrected in `caddy/Caddyfile`
and `scripts/start-stack.bat`.

### Stale claims in the sections below

| claim below | now |
|---|---|
| "All four supervised services" | five (`worker` added) |
| "693 checks" inventory | superseded; run `scripts/run_tests.py` for the current count |
| "Do this first: restart the proxy" | done. The proxy restarted at 19:15 and serves the React dashboard |
| `budget.py` `what_if` omits 147,456 | fixed |
| `budget.py` `_POOL = 131072` fallback | still open (CONSTRAINTS #19) |
| Index state: "one root, three.js only" | 17 package indexes are on disk. The bound-repo index is unchanged |
| "Track B … no code yet" | `web/` exists and is served at `/dash` |
| "`mcp/worker.py` does not exist" | it exists and runs |
| "Wire `domains.py` — still 0 callers" | wired: `proxy`, `selection`, `hints`, `tiers` and others import it |
| `concept_seed` "has zero callers" | called by `fanout` (seeded variants) |
| "Laya is the required router in and out" | Laya is a second signal. It never decides alone, and disagreement escalates |
| "Hemisphere and fan-out are not tools" | still true. `delegate_investigation` is now also off by default |

## There are now two workstreams

They do not block each other and should not be interleaved in one session.

- **Track A, the stack.** The worker loop, Laya, retrieval, benchmarks. This
  is the one with a hard blocker (`mcp/worker.py` does not exist) and it owns
  everything under `mcp/` and `bench/`.
- **Track B, the dashboard.** A React rewrite of the view layer, a component
  library from the design system, and the procedural bonsai. Owns `design/`
  and, when it starts, a new frontend directory. It touches `mcp/server.py`
  only at the very end, to serve static files instead of `PAGE` constants.

Track B cannot break Track A: the dashboard's server is already a JSON API
with a separate HTML view layer, and the Python pages keep serving until the
React build reaches parity.

## Where the stack is right now

All four supervised services answer their `/health` route:

| service | port | state |
|---|---|---|
| llama-swap | 11434 | 200 — `bonsai` ready, `-c 147456`, `-dev CUDA0` |
| proxy | 1234 | 200 |
| tools-api | 1235 | 200 |
| laya | 1237 | 200 |

The watchdog has run ~14h with zero restarts since it was rewritten to check
only `/health` and to restart only the service that failed. Before that rewrite
it asked the proxy for a model named `bonsai` — the proxy advertises only
`yamadori` — and killed the whole stack every ten minutes for hours, costing
33 of 50 benchmark rows.

Every suite passes — **693 checks**, the complete inventory, not a sample:

    ruff check mcp bench scripts --select=E9,F    all checks passed
    mcp/test_tools.py                             356/356
    mcp/test_datasets.py                          131/131
    bench/test_guardrail.py                         92/92
    bench/test_hint_collapse.py                     47/47
    bench/test_laya_calibration.py                  46/46
    bench/test_laya_head.py                         21/21

`bench/test_laya_calibration.py` ends with "docs/LAYA.md still describes the
system", i.e. the doc is asserted by the suite rather than trusted.
`bench/test_guardrail.py` passes while recording that **no policy tested is
usable** — the best still blocks 20+ of 26 adversarial-benign items. That is a
passing test of a failing mechanism, which is the correct shape: the suite
asserts what is true, not what is wanted.

Every touched module imports fresh: `shomen`, `proxy`, `jobs`, `datasets`,
`hints`, `tiers`, `admission`, `dash_data`.

**The one thing not verified here:** an end-to-end generation through the
proxy. Auth is multi-tenant and keys are stored SHA-256-hashed in
`index/accounts/accounts.json` (correct — there is no plaintext to read), so
this has to be run by the operator:

    curl -s http://127.0.0.1:1234/v1/chat/completions \
      -H "Content-Type: application/json" \
      -H "Authorization: Bearer $YOUR_KEY" \
      -d '{"model":"yamadori","messages":[{"role":"user","content":"Reply with exactly: ok"}],"max_tokens":24,"temperature":0}'

## Do this first, before anything else: restart the proxy

The running `mcp/server.py` was started at **02:14** and has been serving
stale in-memory code for about thirteen hours. `server.py` itself was last
written at **13:48**, and `mcp/dash_data.py` at **13:56**.

The visible symptom: `/dash` , `/dash/vitals` and `/dash/results` all answer
200, and **`/dash/data` answers 404** — the dataset manager page. The code is
fine; it was proven in isolation rather than guessed at:

    dash_data.handle_get('/dash/data') -> 200  text/html  39,934 bytes
    registered routes: /dash  /dash/data  /dash/results  /dash/vitals
                       /dash/api/{rest:path}

So the route exists and the running process predates it. A restart also picks
up the `shomen` rename, which the live process has never loaded. The watchdog
supervises `proxy` as a `process` kind and will bring it back on its own if
the restart is done by killing it — that path was tested earlier by killing
tools-api, and only tools-api came back up.

Nothing is in flight: the job queue is empty in every state.

## Two deliberate calls left open in `mcp/budget.py`

Both are one-line changes, both were left alone on purpose because they change
runtime behaviour rather than prose:

- **`_POOL = 131072`** (line ~95), the fallback used when the server cannot be
  reached. It now under-budgets by 16,384 tokens against the live 147,456
  pool. Under-budgeting is the safe direction, which is why it was left, but
  it is stale.
- **`what_if(sizes=(131072, 163840, 196608, 262144))`** (line ~124) lists every
  candidate pool size *except* the one actually running.

The split arithmetic itself was verified exact against the live server, not
assumed: the server reports 147,456, and 88,473 + 36,864 + 22,119 = 147,456.

## Not a bug, checked and cleared

Two `laya_service.py` processes are running and this is **not** the duplicate
-Laya problem from earlier. 29740 is a 4 MB launcher whose parent is the shell;
3040 is its child and holds the 1,899 MB model. One model, one process.


## What changed most recently

`mcp/hemisphere.py` → **`mcp/shomen.py`**. Bonsai lore, like `yamadori` and
`nebari`: the shomen is the front of a tree, the angle it is meant to be seen
from, and finding it means turning the tree and studying every side before the
first cut. Nobody watching the finished tree sees the turning — which is the
module's job. Imports were rewritten across 10 files and all docs paths with
them. The word "hemisphere" survives only in comments and prose, where it is
the right engineering word for a reader.

The **reranker conflict is resolved** in `docs/PLAN.md`: not cut, not trusted,
not used. It is corrupt as deployed (FINDINGS #20 — scores depend on batch
composition, 15–16/89 batched vs 68/89 one at a time), so every reranker
number in the repo is void, including the ones that argued for cutting it.
Nothing is deleted and nothing unmeasured is trusted, which satisfies both the
cut criterion and PROTOCOL rule 9.

## Verified working today, with the evidence

**The hint embedding cache is BUILT and CURRENT.** It was on the pending list;
it is done. Checked properly rather than by the file existing:

    corpus rows 938   cache n=938   mat (938, 1024)
    signature 03f8caa7e65a242a == live signature   MATCH=True
    vector norms min/max 1.0000/1.0000, not all-zero

That last line is the PROTOCOL rule 1 check, and it is there because retrieval
quality was once assessed against an index of all-zero vectors and a component
was cut on the result.

**Selection and abstention both behave.** On "static array, many range-sum
queries" the top hint is `prefix sums` at 0.815. On "what is the weather in
paris today" it returns **nothing**, which is the correct outcome and the
whole point of a similarity floor rather than a rank cutoff.

**It also reproduced the bucketing gap live, on the first try.** That same
range-sum query returned three hints, and the third was *"Point updates
interleaved with range sums: Fenwick tree"* — the mutually exclusive
alternative to the first one. Two hints arguing with each other in front of
the model at code-writing time, which is the harm `mcp/hints.py` is written to
avoid. The deciding fact (*are there updates?*) is buried in prose and is only
a discriminator relative to its sibling. This is `docs/ROADMAP.md` 1.6, it is
the highest-leverage open door, and it is now demonstrated rather than
predicted.

**Index state.** `index/code.sqlite3` holds 7,742 chunks / 5,874 defs /
163,903 refs across **one root** — three.js only. `index/corpus.sqlite3` has
1,831 events, `index/nebari.sqlite3` 60 sessions. Most target domains still
have no source to retrieve, which blocks any domain benchmark.

**Lint is clean across the whole Python surface** — `mcp`, `bench` and
`scripts` — including the three long-standing unused imports in
`bench/laya_before_after.py` and `bench/laya_grounded_contrast.py`, which were
confirmed genuinely unused before removal (`stratified_split` looked unused to
a grep but is called at line 119, so it stayed).


## The ADVERTISED hop budget is deleted (FINDINGS #27)

It was a fossil. Both mechanisms — the budget stated in the prompt and the cap
in the loop — were built in response to a model that looped on tools returning
**empty strings**, which it could not tell apart from "nothing matched". That
root cause was fixed separately with structured tool errors; the budget
machinery was never revisited.

It was also false: both proxy loops read `MAX_TOOL_HOPS` and never the tier,
so a `low` request was told "at most 4 tool-calling turns" and actually had
twelve.

Removed: `budget_line()`, `augment_messages(hops=)`, the `BUDGET:` paragraph
and the "one searching turn left" warning in `shomen.py`, and the `hops` key
from every tier. Kept: hop **counting** for telemetry, and `MAX_TOOL_HOPS` /
`MAX_HOPS` (8 → 16) as **unadvertised runaway breakers** protecting the two
GPU lanes. Nothing tells the model about them; tripping one is a defect
report, and `shomen`'s failure message now says so.

The tier ladder is now purely cognitive effort: `minimal` → `low` (search) →
`medium` (+hints) → `high` (+fan-out) → `max` (+deep thinking).

**One thing to re-measure because of this:** the context-economy ratio (0.29x
main-context, 26/26, p=2.98e-08) was taken with both arms on an identical
8-hop budget. The comparison stays valid — the same budget cannot explain a
difference between arms — but the magnitude may move now that nothing steers
the deep-thinking context to stop early. Direction holds, number is stale.


---

## Track B: the dashboard and the bonsai (design done, no code yet)

Everything needed to start is in `design/`. Nothing has been built.

| file | what it is |
|---|---|
| `design/README.md` | **read first** — three traps in the source material |
| `design/DESIGN.md` | the design system: token frontmatter + prose rationale |
| `design/STACK.md` | stack decision, versions read from npm 2026-09-22 |
| `design/BONSAI-VIZ.md` | the 3D bonsai plan, including the latent-space axis |
| `design/mockups/*.html` | three reference mockups, desktop / mobile / nebari |

### The three traps, because each fails quietly

1. **`DESIGN.md` ships two palettes that disagree, and the prose one is
   stale.** Seven of the eight colours named in its "Colors" prose appear
   **zero times** in any mockup. Build from the **frontmatter**. The prose is
   still good for rationale, never for hex.
2. **Every number in the mockups is fabricated** — 4x RTX 4090, 96 GB, a 70B
   model, 8.4M embeddings. None of it is this stack. Wire panels to
   `/dash/api/vitals`, never to mockup values.
3. **`TAPROOT` / `BRANCH` / `SHOOT` and `canopy` / `rootstock` / `graft`**
   appear in the design and in `AGENTS.md`'s naming table but **not** in
   `config.yaml`, which says `primary` / `retrieval` / `ondemand`. Aspirational.
   Do not "fix" either side without deciding which one moves. **Needs an
   operator decision.**

### Stack, with the version traps

Two viable paths, both resolving with no overrides. `design/STACK.md` has the
full constraint graph and recommends **Path B**.

    Path A (stable)              Path B (alpha, recommended)
    react 19.3.0                 react 19.2.8          <- NOT 19.3
    three 0.186.0                three 0.186.0
    fiber 9.8.0                  fiber 10.0.0-alpha.5
    drei 10.7.8                  drei 11.0.0-alpha.7
    @react-three/postprocessing  postprocessing 6.39.5 <- the plain one
      3.1.2

Shared by both: `vite` 8.3.0, React Compiler 1.0.0, `react-fate` 1.6.0,
`@stylexjs/stylex` + `@stylexjs/postcss-plugin` 0.19.1.

- **r3f v10 alpha and drei v11 alpha are real and current** — fiber canaries
  published 2026-09-20. An earlier draft of `STACK.md` claimed "there is no
  v10"; that came from reading `dist-tags` and short-circuiting on `beta`
  before reaching `alpha`, and it is corrected in place.
- **The two paths want different React versions.** Path B peers
  `react >=19.0 <19.3`, so 19.2.x, newest 19.2.8. Path A wants 19.3.0. This is
  the thing most likely to be got wrong when switching between them.
- **`@react-three/postprocessing` 3.1.2 shipped 2026-09-22** and is current;
  there is no v4 and no replacement package. Its peer `fiber >=9.7.0` will not
  formally match a `10.0.0-alpha` prerelease, but that is a resolver complaint,
  not a runtime one — pnpm/yarn/bun warn, npm wants one `overrides` line. Plain
  `postprocessing` 6.39.5 is the alternative and fits the custom passes the
  showpiece needs anyway.
- **`postprocessing` v7 exists but is the OLDER branch** — `7.0.0-beta.16`
  (2026-02-19) needs `three <0.184.0`, which cannot intersect r3f 10's
  `>=0.185.0`. The v6 line is what ships: 6.39.5 (2026-09-09) raised its cap to
  `<0.187.0` specifically for three 0.186.
- **three is boxed into 0.185.0-0.186.0** by `postprocessing`, and r3f 10 needs
  `>=0.185.0`. Usable window is 0.185.0-0.186.0, and three ships monthly.
- **Use `@stylexjs/postcss-plugin`, not `vite-plugin-stylex`** (0.13.0, Nov
  2024, two years behind core).

### Serving

Prod: `mcp/server.py` serves committed `dist/` at the site root. Dev (`--dev` /
`YAMADORI_DASH_DEV=1`): does not mount `dist`, and the browser talks to Vite
on :5173 with Vite proxying `/dash/api/*` to :1234. **The proxy points Vite to
FastAPI, never the reverse** — relaying HMR's websocket through Starlette
degrades to silent full-page reloads. `base: '/'` in the vite config, SPA
fallback registered after every other route, and the watchdog stays at four
services.

`dist/` is committed so zero-config holds — Node is a contributor dependency,
never a runtime one. It needs a `dist/.buildinfo` source-hash check in the test
suite, or a forgotten rebuild silently ships a dashboard that does not match
its own source.

### Day one, before any tree code

A spinning cube with React Compiler **on**, plus one component mutating a ref
in `useFrame` and one deriving geometry in render. The compiler assumes
purity; r3f's idiom is mutation. It mostly works — but find out while the
answer is still "drop the compiler" rather than "rewrite the visualisation".

### The bonsai, in one paragraph

A procedural 3D bonsai that *is* the telemetry. **Fan-out is the branching**:
N samples are N limbs, each seeded by its own `concept_seed` word (that module
has zero callers and this gives it its first), the chosen sample thrives, the
culled ones bleach to `shari` and stay. Interpolation works because topology
is a **fixed superset** and only parameters animate, so nothing ever pops.
The latent axis is real and was probed live: the model returns `logprobs` with
`top_logprobs: 5` including token ids, **on the reasoning stream** — entropy,
margin and surprise per token, with the branch growing along the trajectory
through a **frozen** PCA of `token_embd.weight`. What it is *not* is the
residual stream: `/embedding` on bonsai returns **501**, and enabling it means
a second 27B on a card we do not have.


## The first thing the next run must build

**`mcp/worker.py` does not exist.** `mcp/datasets.py` enqueues
`dataset.fetch` (line 390) and stage jobs (line 490) into the durable queue,
and *nothing ever claims them*. The queue is currently empty, so nothing is
stuck — but the moment a dataset is submitted from `/dash/data`, its job sits
in `queued` forever and the dashboard shows a pipeline that never advances.

The whole dataset pipeline is blocked on this one file. It needs:

- a claim loop per lane, honouring `LANES = {gpu: 1, cpu: 4, net: 4}` — the
  gpu lane is 1 because two jobs on one card is not slow, it is wrong
- `jobs.beat()` from inside long handlers, or a 450-second generation looks
  identical to a dead worker and `reclaim()` hands its work to someone else
- `jobs.reclaim()` on startup
- five stage handlers: `dataset.fetch` (net), `dataset.extract` (gpu),
  `dataset.index` (gpu), `dataset.label` (cpu), `dataset.train` (gpu) — the
  mapping is `datasets.py:90-98`
- a handler that raises must land in `errored`, never `done`. That distinction
  is why `jobs.py` exists: a resume path that counted errored rows as done
  once poisoned a benchmark file permanently.
- tests in the style of `mcp/test_datasets.py`, against a temp DB

## Then, in order

1. **Laya reconsideration** — spec-shaped input first (this bumps `prompt_rev`
   and invalidates cached features), then training, against embeddings on the
   same normalised input. Four predictions from the architecture have now been
   wrong; treat a fifth as needing measurement, not argument.
2. **Wire `domains.py`** — still 0 callers. Cheapest real win: stops tools
   being offered where nothing is indexed.
3. **Build buckets of mutually exclusive alternatives** and collapse each to
   one member, so hints cannot argue. Demonstrated live above. The hint
   embedding cache this used to be blocked on is already built.
4. **ROADMAP open doors** 1.2 (conceptual query set), 1.3 (reranker, only
   after the batching fix), 1.4 (independent ground truth), 1.5 (hint
   selection vs `fixed`), 1.6 (data shaping — the highest-leverage unknown).
5. **Combined benchmark**, all four augmentations on, in the target domains —
   not LiveCodeBench, which has no repository and nothing to retrieve.

## Standing rules that were bought with mistakes

- Nothing is cut until the composed systems are built, working and
  benchmarked. Single measurements do not get to kill a component.
- The second brain is called **thinking** or **deep thinking** in anything a
  user sees. Bonsai lore for internal names. Never "second context",
  "hemisphere" or "investigating in a second context" on the wire.
- Hemisphere and fan-out are not tools. They are classified selectors we run
  automatically; Laya is the required router in and out.
- A tool result returns the truth as structured data: the situation, whether
  it is retryable as a fact, and a remedy with an owner. Never a hint to try a
  different tool.
- A hint must never harm. Emitting nothing is a correct outcome.
- Don't detect, instruct: the injection *screen* carried zero information
  (φ = +0.000) and was deleted; the hardened data/instruction paragraph in the
  prompt cut injections ~10× (p = 0.0021) and was kept.
