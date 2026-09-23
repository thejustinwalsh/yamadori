# Dashboard stack

Decided 2026-09-22. Every version below was read from the npm registry on that
date, not recalled.

> **Correction, same day.** An earlier version of this file said "there is no
> r3f v10". That was wrong. It came from reading `dist-tags` and short-circuiting
> on `beta` before reaching `alpha` — the alpha was in the same response and the
> check discarded it. A probe that looked like coverage and provided none, which
> is the failure class `docs/PROTOCOL.md` rule 1 exists for. The alphas are real
> and current: r3f canaries were published **two days** before this was written.

## The choice: stable or alpha

Both resolve cleanly. They are different bets, not better and worse.

### Path A — stable

    react / react-dom            19.3.0
    three                        0.186.0
    @react-three/fiber           9.8.0
    @react-three/drei            10.7.8
    @react-three/postprocessing  3.1.2

Everything peer-resolves with no overrides, and `@react-three/postprocessing`
gives selective bloom and god rays as R3F components.

### Path B — alpha (r3f 10 / drei 11)

    react / react-dom            19.2.8      <- NOT 19.3
    three                        0.186.0
    @react-three/fiber           10.0.0-alpha.5
    @react-three/drei            11.0.0-alpha.7
    postprocessing               6.39.5      <- the plain one, not @react-three/

| package | published | tag |
|---|---|---|
| `@react-three/fiber` 10.0.0-alpha.5 | 2026-09-08 | `alpha` |
| `@react-three/fiber` 10.0.0-canary.24d1f81 | **2026-09-20** | `canary` |
| `@react-three/drei` 11.0.0-alpha.7 | 2026-09-05 | `alpha` |

This also resolves with **no overrides**, but only because of the
postprocessing substitution below. drei 11 alpha is `"type": "module"`, pure
ESM.

---

## Both paths share these versions

`vite` 8.3.0, `babel-plugin-react-compiler` 1.0.0, `react-fate` 1.6.0,
`@stylexjs/stylex` + `@stylexjs/postcss-plugin` 0.19.1. Optional
`@nkzw/fate-indexeddb` 1.6.0.

---

## Four traps, each of which fails quietly

### 1. On Path B, React is pinned to 19.2.x — and 19.3 breaks it

    @react-three/fiber 10.0.0-alpha.5   peer: react >=19.0 <19.3
    @react-three/drei  11.0.0-alpha.7   peer: react >=19.0 <19.3
    react-fate 1.6.0                    peer: react ^19.2.0

Intersection is **exactly 19.2.x**. Newest is **19.2.8** (2026-07-21). Pin it.

On Path A the window is `>=19.2 <19.4`, so 19.3.0. **The two paths want
different React versions**, which is the single thing most likely to be got
wrong when switching between them.

### 2. `@react-three/postprocessing` peer-resolves against r3f 10, but only formally

Latest is **3.1.2, published 2026-09-22** — actively maintained, and there is
no v4, no alpha tag, and no replacement package in the registry.

Its peer is `@react-three/fiber >=9.7.0`. By npm semver a prerelease
(`10.0.0-alpha.5`) does not satisfy that, because prereleases only match ranges
carrying a prerelease comparator at the same version tuple.

**This is a resolver complaint, not a runtime one.** pnpm, yarn and bun warn
and continue; npm needs `--legacy-peer-deps` or a one-line `overrides` entry.
The library itself is very likely fine against r3f 10 — anyone already running
this combination would simply not have noticed the range.

An earlier draft of this file said the package "cannot get one by override".
That was wrong and overstated: an override is the ordinary fix and it works.

**Still worth knowing there is an alternative.** Plain `postprocessing` 6.39.5
is framework-agnostic ESM peering only on three, so it sidesteps the range
entirely. That matters here less as a workaround than as a fit: the showpiece
needs selective bloom masked to emissive branches and an occlusion-masked
god-ray pass, which is custom work either way, and owning `EffectComposer`
through `useThree` is about thirty lines. Use the wrapper if it works; reach
for the plain library when a pass needs to be hand-written.

### 3. three is boxed into 0.185.0-0.186.0, and `postprocessing` v7 cannot come

`postprocessing` has three live dist-tags, not one:

    latest  6.39.5          2026-09-09   three >= 0.168.0 < 0.187.0
    beta    7.0.0-beta.16   2026-02-19   three >= 0.179.0 < 0.184.0
    alpha   7.0.0-alpha.4

**v7 is the older branch, not the newer one.** The v6 line is what is actively
shipping: 6.39.4 capped three at `<0.186.0` and **6.39.5 raised it to
`<0.187.0`**, which is the three-0.186 support bump. v7 beta has not moved
since February.

**v7 cannot be used with r3f 10.** It requires `three < 0.184.0`; r3f 10 alpha
requires `three >= 0.185.0`. The ranges do not intersect. If a v7 build is
wanted, three has to drop to 0.183.x and r3f 10 is off the table.

So the usable window is:

    r3f 10 alpha         three >= 0.185.0
    postprocessing 6.39.5  three < 0.187.0
    ------------------------------------------
    three 0.185.0 - 0.186.0,  currently 0.186.0

**One minor of headroom, and three ships roughly monthly.** When 0.187 lands,
`postprocessing` has to bump before three can. Pin three exactly and treat a
three upgrade as a coordinated change, not a routine bump.

### 4. Use StyleX's own PostCSS plugin, not the community Vite plugin

    @stylexjs/stylex          0.19.1   published 2026-09-15
    @stylexjs/postcss-plugin  0.19.1   published 2026-09-15
    vite-plugin-stylex        0.13.0   published 2024-11-06

`vite-plugin-stylex` is nearly two years behind a core library that shipped a
week ago, and StyleX's compiler output format is not frozen across that gap.
The official PostCSS plugin versions in lockstep with core and runs under Vite.

---

## Recommendation

**Path B**, for three reasons that are about this project specifically:

- The showpiece needs hand-written passes anyway. §2's substitution is not a
  workaround here — selective bloom masked to emissive branches plus an
  occlusion-masked god-ray pass is custom work in either path, and owning the
  composer directly makes it easier, not harder.
- r3f 10 is where the React 19 concurrent story is actually being worked out,
  and this app is being written now against React Compiler. Starting on 9.8.0
  means porting later, during a migration whose alpha is already six months
  old.
- `three >=0.185.0` on r3f 10 matches the version already indexed in this repo
  (`three@0.185.1`), so retrieval about three.js is answering from the same
  major the code is written against.

**Take Path A instead if** the canary churn is unwelcome — r3f 10 published
canaries on 2026-09-07, -08, -13, -16 and -20, so it is moving weekly and an
alpha can break between two of them.

---

## The risk to test on day one, not at the end

**React Compiler and r3f have a philosophical disagreement.** The compiler
assumes components are pure and memoizes on that basis. r3f's entire idiom is
mutating three.js objects — `useFrame` writing to `ref.current.position`,
imperative material updates, mutable scene graph.

This mostly works, because mutation inside `useFrame` happens outside render.
But it is exactly the kind of interaction that produces a subtle, intermittent
failure discovered three weeks in, and the fix at that point is "turn off the
compiler", which loses the reason for having a build step at all.

So: **Phase 0 is a spinning cube with the compiler on**, plus one component
that mutates a ref in `useFrame` and one that derives geometry in render.
Confirm the compiler's output is correct before any of the tree is written. If
it is not, that is worth knowing while the answer is still "drop the
compiler" and not "rewrite the visualisation".

`babel-plugin-react-compiler` has an opt-out directive per file, so a partial
answer is available: compiler on everywhere except the r3f subtree.

---

## Data layer

**Not adopted (2026-09-22).** `react-fate` 1.6.0 exists and its peers
(`react ^19.2.0`) accept 19.2.8, but it is the wrong model for these
endpoints. Its `./vite` plugin generates the client from a TypeScript server
module (`transport: 'graphql' | 'native' | 'trpc' | 'void'`); ours is Python.
Hand-built, its cache normalises entities by `__typename` + `id`
(`getId` throws "Missing 'id' on entity record") behind masked, per-field
views, and errors surface by throwing to Suspense/error boundaries. The
payloads here are id-less snapshot documents keyed by URL, polled on a timer,
whose typed failures (401, stale age) each panel prints. Its own README says
alpha, not production ready.

What is used instead: `web/src/api/cache.ts`, one snapshot per `/dash/api`
path with in-flight de-duplication, which `usePoll` reads and writes; and
`web/src/routes.ts`, which a nav link (`ui/NavLink.tsx`) calls on hover,
focus and press to start the screen's chunk and data. Routing is wouter
3.11.0.

The four endpoints already exist and are unchanged by any of this:

    /dash/api/vitals      GPU, processes, listeners, endpoints, context pool
    /dash/api/datasets    dataset list
    /dash/api/dataset     one dataset
    /dash/api/results     benchmark results

This is the part that makes the rewrite cheap: the dashboard's server is
already a JSON API with a separate HTML view layer. React replaces the view,
not the server. The Python pages (`mcp/dash_*.py`, 3,481 lines) stay serving
until the React build is at parity, then the FastAPI routes in `mcp/server.py`
serve static assets instead of `PAGE` constants.

---

## Component library

Extracted from `design/DESIGN.md`'s **frontmatter** tokens — the prose palette
is stale, see `design/README.md`. StyleX's `defineVars` is a direct match for
the token frontmatter, so the design system becomes a typed TS module rather
than a CSS file, and a token that does not exist becomes a type error.

The mockups use zero CSS variables — everything is hardcoded hex. Extracting
those into tokens **is** the component-library work, not a preliminary to it.

---

## Distribution: prebuilt assets are committed, so zero-config holds

A build step does not reach the user. **The built bundle is committed to the
repo**, `mcp/server.py` serves it as static files, and someone self-hosting
this clones it and runs Python exactly as they do today. Node is a
*contributor* dependency, never a *runtime* one.

That is the whole argument, and it holds. There is no npm install in the
install path, no node process beside the Python one, and nothing new to
supervise — the watchdog still has four services to check.

### The one discipline it requires

Committed build output goes stale silently. Someone edits a `.tsx`, forgets to
rebuild, commits, and the repo now ships a dashboard that does not match its
own source. Nothing errors. The dashboard just quietly is not what the code
says it is.

That is the same failure shape as everything else this repo has been bitten
by: a health check that knew too much, an index of all-zero vectors, a budget
that stated a number nothing enforced. **A thing that lies without failing.**

So the build output gets the same treatment those got — made to fail loudly:

- A `dist/.buildinfo` recording the content hash of every source file that
  fed the build.
- A check that recomputes those hashes and **fails** if the committed bundle
  does not match the committed source. Runs in the test suite, alongside the
  other 693 checks, so it is not a separate thing to remember.
- The check needs no Node. It hashes files and compares, which is Python and
  milliseconds. A contributor without Node installed can still be *told* the
  bundle is stale, even though they cannot rebuild it.

Rebuilding stays a documented two-command step for whoever changes the
frontend. Everyone else, including every user, never knows it exists.

---

## Two serving modes, one flag

Default is production: `mcp/server.py` serves the committed `dist/` at the
site root (`/`; old `/dash/*` links 302 there). With a `--dev` flag (or `YAMADORI_DASH_DEV=1`) it does
not mount `dist` at all and defers to Vite's HMR server.

### The proxy points from Vite to FastAPI, not the other way round

This is the detail that decides whether HMR actually works.

In dev the **browser talks to Vite** on :5173, and Vite's `server.proxy`
forwards `/dash/api/*` through to FastAPI on :1234. One origin, so no CORS,
and HMR's websocket is a direct Vite connection.

The tempting inversion — browser hits :1234, FastAPI proxies through to Vite —
keeps a single port but has to relay Vite's HMR websocket through Starlette.
That path is fragile, and when it degrades it does so by silently falling back
to full page reloads, which is the worst outcome: HMR appears to work and
quietly is not. Same failure shape as everything else in this file.

So: **dev is `localhost:5173/`, prod is `localhost:1234/`.** Two URLs
is a smaller cost than a websocket relay. `--dev` makes the Python side print
the Vite URL rather than serving a stale bundle, so nobody stares at an
unchanging page wondering why their edit did nothing.

### Three details that are easy to miss

- **`base: '/'`** in `vite.config`, matching where the server serves it
  (chunks at `/assets/`). Getting this wrong produces a blank page and 404s on
  every chunk, in prod only — it works fine in dev, which is how it ships.
- **SPA fallback.** Any unknown GET that asks for `text/html` returns
  `index.html` so client-side routes survive a refresh. It is registered
  after every other route and never answers `/v1`, `/health`, `/dash/api`,
  `/dash/classic` or `/assets`; a client that does not ask for HTML gets a
  404 (and `/` stays the JSON descriptor for it).
- **The watchdog does not learn about Vite.** It supervises four services by
  `/health`, and a dev server is not one of them. A developer's Vite process
  dying is a developer's problem; the watchdog restarting it would be the
  health check knowing too much all over again.
