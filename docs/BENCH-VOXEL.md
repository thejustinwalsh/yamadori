# The voxel-scene benchmark (bench/voxel)

One prompt people post online, sent verbatim through the proxy on `:1234`.
The answer is a single HTML file, which a sandboxed browser measures. With
`--task r3f-stack`, the same prompt with its last sentence swapped for the
operator's stack sentence; the project the model writes is built before the
browser measures it.
The operator judges the result from a static page. No model grades it, and
nothing the check finds is ever sent back to a model: graders only measure.

## The task

Sent verbatim as the only user message. There is no system prompt, no tools,
no temperature, no `max_tokens` and no follow-up. `bench/voxel/run.py`
`PROMPT` holds it, and `test_voxel.py` pins its sha256
(`2cee15c5...22fe8c4`):

> Design and create a very creative, elaborate, and detailed voxel art scene
> of a pagoda in a beautiful garden with trees, including some cherry
> blossoms, add a village with people living in it. Make the scene impressive
> and varied and use colorful voxels. Make it really detailed use as many
> voxels as you want, we have a power machine here. Create a single HTML file.

The prompt is public, so it may be in training data. Any gain an arm shows
on it is a gain on a seen prompt.

## The stack option: `--task r3f-stack`

A second task beside the first, not a second tool. `single-html` (the
verbatim prompt above) stays the default and is unchanged, sha256 included.
`--task r3f-stack` asks for the same scene built with the operator's stack.

The prompt is the verbatim prompt above with ONE change: its last sentence,
" Create a single HTML file.", becomes the operator's sentence (operator,
2026-09-27: "So I just said build the pagoda and octopus-invaders with r3f
(react-three-fiber) v10 and Koota and pmndrs math." ... "That changes the
prompt."):

> ... we have a power machine here. Build it with r3f (react-three-fiber) v10
> and Koota and pmndrs math.

Nothing else of ours is added: no pinned versions, no project layout, no
stack rules, no build gate and no delivery convention. The model decides the
project, and the extraction takes whatever layout it writes. `run.py`
asserts the last sentence is there before replacing it; `STACK_SENTENCE` is
the same string as the Octopus V4 prompt's (`variants.V4_TECH`).

`test_voxel.py` pins its sha256 (`a5d46abd...fb70407cd`). The first
version (2026-09-26, `9a4d5113...8a8859a2d`, 493 words) appended our V4 pins,
`variants.stack_rules("scene state")`, a required layout, a build gate and a
`File: <path>` convention; it wrote the model's architecture for it and was
never run.

**A tag holds one task.** The manifest records the task, the prompt and its
sha256 (for r3f-stack, also the stack sentence and the sentence it
replaces). Every summary row records the task and the sha256 of the prompt
actually sent. `run.py` refuses a rerun of a tag with another `--task`, or
with a prompt whose sha256 changed, before sending anything. A manifest or
row with no task predates the option, so it is read as single-html.

The stack sentence is the operator's; the rest of the prompt is public, so it
may be in training data.

## Arms

An arm is three things:

- a tier: the body's `reasoning_effort`
- an optional `X-Yamadori-Features` header, which forces everything of ours
  off to make an effort-matched control
- a model id: `ARM@MODEL`, for a sibling model once one is configured

The proxy resolves an unknown model name to `yamadori` without saying so.
Preflight therefore refuses any model that `/v1/models` does not list.

| arm | body | header | status |
|---|---|---|---|
| `xhigh` | `reasoning_effort: xhigh` | none | **the planned arm, and the default** |
| `max` | `max` | none | kept for later |
| `low` | `low` | none | kept for later: the model as it ships |
| `xhigh-off` | `xhigh` | `retrieval, skills, investigate, check_code, repair` all false, `fanout` 1 | kept for later: the effort-matched control of `xhigh` |
| `max-off` | `max` | the same | kept for later |

**Operator, 2026-09-26:** "our xHigh thinking is all I care to test anymore,
until it actually works." The runner's default is `xhigh` alone. The other
arms run only when `--arms` names them.

Two notes for a later comparison. `low` and `xhigh` both send the template
`medium` thinking (the `tiers.TIERS` table in AGENTS.md), so `low` vs
`xhigh` is already matched on thinking effort. `xhigh-off` is the stricter
control, because it keeps xhigh's tier caps and switches.

## Power

At n=3 reps of one arm there is no comparison, only a description. The page
shows what three samples of one arm look like. Two arms at n=3 on one
prompt can show only a difference the operator can see in every rep.
Nothing here is a finding until it repeats (PROTOCOL rules 4 and 10).

## How to run

Use the stack interpreter,
`C:\Users\jwals\textgen\installer_files\env\python.exe`. Run one GPU
consumer at a time: fix, test, deploy, then run.

```
# 1. generate: xhigh, 3 reps (the first run)
python bench/voxel/run.py --key-file PATH --tag vx-xhigh-1

# 2. measure every page in the sandboxed browser (CPU only; after step 1)
python bench/voxel/check.py --tag vx-xhigh-1

# 3. the page to judge: bench/voxel/results/vx-xhigh-1/index.html
python bench/voxel/compare.py --tag vx-xhigh-1
python bench/voxel/compare.py --tag vx-xhigh-1 --blind   # when there are 2+ arms

# the stack option: its own tag (a tag holds one task)
python bench/voxel/run.py --key-file PATH --tag vx-stack-1 --task r3f-stack
python bench/voxel/check.py --tag vx-stack-1     # build, static checks, then the browser
python bench/voxel/compare.py --tag vx-stack-1
```

Other forms:

- `--preflight-only`: checks without sending anything.
- `--list-arms`: prints the arms.
- `--reps 5` on the same tag: adds reps 4 and 5.
- `--arms xhigh,xhigh-off`: an effort-matched pair.
- `--max-tokens N`: sends an answer allowance. It is recorded, and it is not
  what a chat client sends.
- `check.py --html FILE --out DIR`: checks one page. The two fixtures are
  hand-written, known-answer smoke pages.

## What `run.py` does

- **Preflight.** It refuses to start in three cases:
  - Another GPU consumer is running. The check uses the `BUSY` list and
    `busy_processes()` of `bench/octopus/run.py`, and that list now names
    this runner too.
  - A llama-server slot is processing on either of two reads 5 s apart.
  - The proxy or the model list does not answer.

  The busy-process check runs again before every rep. If it trips, the run
  stops, and rerunning the same tag resumes it.
- **Not run is not a result.** A 429 or a refused connection is waited on
  for up to `--busy-max-s` (15 min), then recorded as `not_run`. A rerun
  retries it.
- **Stack errors are never model outcomes (PROTOCOL rule 3).** These are
  `http_error`, `exception` and `mismatch`. A `mismatch` means
  `x_yamadori.tier` is not the arm's tier, or the request was served as a
  utility call. `--retry-errors` reruns stack errors.
- **Finished outcomes.** `ok` and `length` count as done. `length` means the
  reply was cut, and extraction flags an unclosed fence.
- **Extraction (single-html).** It takes the largest fenced block that is a
  page. Failing that, it takes a raw `<!doctype html>`/`<html>` ... `</html>`
  span. It never repairs anything. It records how many blocks there were and
  in which languages, whether the fence closed, and whether `</html>` is
  present.
- **Extraction (r3f-stack), `extract_project`.** Every fenced block gets its
  path from the first of these that names one:
  1. the convention line (`File: <path>`, the nearest non-blank line before
     the fence; bold or backticks around it are tolerated, and exact uses
     are counted)
  2. the fence's info string (```` ```tsx src/App.tsx ````,
     ```` ```tsx:src/App.tsx ````, `title="..."`, or a lone path)
  3. a heading, bold or backticked path on the line before the fence
  4. a first-line comment naming the path (`// src/App.tsx`, `/* */`,
     `<!-- -->`)

  The prompt asks for no convention; models often write a `File:` line or a
  heading unasked. When no block is named `index.html`, the largest UNNAMED
  block that is an HTML page becomes `index.html` (source `page`): a reply
  that is one page with an import map is a project of one file.

  Fences nest. A README's own ```` ```bash ```` blocks stay inside it. A block
  left open when the next `File:` line starts a new fence ends there and is
  flagged as unclosed. Paths are normalised (forward slashes, no `./`). An
  absolute path, a drive, `~`, `..` or a character a file name cannot hold
  is refused and recorded, never written. A repeated path keeps its last
  block and is recorded. Blocks with no path are counted. Files are written
  byte for byte under `reps/<rep>/project/` (gitignored). The row records the
  counts by source, the problems, the layout (`npm`: a package.json at the
  project root; `static`: an index.html and no package.json; `none`), and
  the project root. The root is `""`, or the folder holding `package.json`
  when a reply wrote `pagoda/...`. Nothing is moved, and nothing is repaired.
- **The key.** It is read from `--key-file` or `YAMADORI_TEST_KEY`. It is
  sent only as a header, and never printed or written.
- **Timeout.** The default is 4 h. The request blocks, and the main share is
  about 132K tokens at about 9 tok/s at worst.

## What `check.py` measures

Chromium runs in the pinned Playwright image
(`bench/octopus/playwright.Dockerfile` becomes `octo-playwright:1.63.0`, one
pin shared with the Octopus grader). It runs on a per-page `--internal`
Docker network whose only way out is the egress gate
(`bench/sandbox/sandbox_net.py`, prefix `voxel`). The gate reaches only
global addresses on 80/443, never the Windows host.

The page is served from a directory that holds nothing but `index.html`.

On top of the gate, a request route admits only `CDN_HOSTS` in
`page_check.py`. This is a choice: jsdelivr, unpkg, cdnjs, esm.sh, skypack,
jspm, esm.run, threejs.org, babylonjs, and Google Fonts. The route aborts
everything else and records it. Each row therefore says:

- whether the page needed the network (`needs_network`)
- which CDN hosts it loaded from (`cdn_hosts`)
- what was blocked (`blocked_hosts`)
- what the gate itself let through (`gate.allowed_hosts`, from the gate's
  log)

The same inputs happen at the same times for every page:

1. Load, allowing 90 s.
2. First draw, allowing 30 s.
3. Default view at 5, 10 and 15 s after load: shots `t05`, `t10`, `t15`.
4. fps over t05 to t15, with no input.
5. Stats.
6. `t16`, taken 2 s later with no input. This is the motion baseline.
7. A drag of 1/8 width to the right: `orbit_right`.
8. A drag of 1/12 height down: `orbit_high`.
9. Five wheel ticks: `zoom_out`.

`Math.random` is seeded (mulberry32, seed 1), so a page builds the same
scene on every check. WebGPU is hidden, as in the Octopus grader, so three.js
falls back to WebGL2.

| measure | how | limits |
|---|---|---|
| console / page errors | Playwright events | the first 4 texts are shown |
| non-blank | PIL on each shot: blank means at least 99% of pixels are within 8 levels of the modal colour, or the mean channel std is under 2 (a choice) | |
| fps | the page's own requestAnimationFrame rate, t05 to t15 | **software rendered** (SwiftShader), so it ranks pages against each other, not against a real card |
| voxels | three.js scene: `BoxGeometry` meshes plus `BoxGeometry` InstancedMesh instances | merged geometry is not a box: it shows as `cube_equiv_nonbox` = non-box triangles / 12, an estimate |
| colours | three.js: distinct material colours, instance colours and sampled vertex colours (8-bit). Screen: 5-bit colours covering at least 0.01% of pixels in t15 | capped at 4,096 |
| extent | bounding box of the voxels, and of everything (a sky dome or ground plane dominates the second) | at most 500K instances scanned |
| orbit | the three.js camera matrix moved by a drag more than 3x its no-input change. With no three.js camera, the pixel change must exceed 3x the no-input change and 2 levels (a choice) | pointer-lock or keyboard-only controls read as no orbit |
| WebGL | draw calls, instanced calls, instances and primitives per frame, from hooks on the WebGL context. This works for any page | |

three.js is found through its own `__THREE_DEVTOOLS__` hook, which covers
module and global builds. A page without three.js still gets errors,
non-blank, fps, WebGL counters, network and pixel-orbit readings.

### r3f-stack: build, static checks, then the same sequence

For an r3f-stack rep, `check.py` does three things. Nothing it finds goes
back to a model.

1. **Build**, in the layout the model chose (`grade.build_mode`).
   `bench/octopus/serve_app.sh` runs in the Octopus grader's pinned node
   image (`bench/octopus/run.py` `IMAGE`, by digest), on a `voxel` sandbox
   network that reaches npm through the allow-public-only gate, on a copy:
   - a `package.json`: `npm build` -- npm install, `tsc --noEmit` only when
     the project has a `tsconfig.json` (else `tsc.rc` is `skip`: the prompt
     names no language), `npm run build`; `dist/` is copied out.
   - no `package.json`: `js build` -- the project itself (its `index.html`
     at the root) is the page.
   `check.py` records the exit codes, the `error TS` counts, the first TS
   error lines, the seconds, `npm ls` and the dist size, with the log tails
   in `check.json`.
   - A build that ran and failed is a **measurement**: `build.outcome` is
     `failed`, and there is no page check.
   - A build that could not run is an `error`: Docker, the gate, or the
     45-minute container limit. It writes no `check.json`, so a rerun
     retries it.
2. **Static stack checks** -- what the sentence asks, nothing more.
   `bench/octopus/grade.py` is loaded by path, and its V4 stack checks run
   over `grade.Src(project)` (the tree-sitter parse), with the build's
   `npm ls`. While `grade.py` loads, the Octopus `run` module stands in for
   this directory's `run`. The checks (grader v4):

   | check | what it looks for |
   |---|---|
   | `r3f_v10_canvas` | `@react-three/fiber` major 10 -- the version npm installed, else what package.json or an import-map URL declares (v10 exists only as `10.0.0-alpha.*` / canary; `latest` is 9.8.1) -- and `Canvas` imported from `@react-three/fiber` (any subpath) and used |
   | `koota_world_traits_queries` | koota imported; `createWorld` and `trait` called; a query called |
   | `math_used` | `math` or a `math/*` subpath imported (a bare specifier or a CDN URL), and a name it imports used outside the import. `Math.random` calls are counted |

   The 2026-09-26 checks `webgpu_canvas`, `tsl_node_material` and
   `package_pins_exact` are gone: the prompt no longer asks for a WebGPU
   canvas, TSL or pins. These checks detect the stack's presence. They do
   not show that the whole scene uses it.
3. **Page check.** The built `dist/` goes through the same `page_check.py`
   sequence as a page. `dist/` is served as a directory, so `/assets/*.js`
   resolves locally.

   Not verified: whether three's `WebGPURenderer` announces itself to
   `__THREE_DEVTOOLS__` the way `WebGLRenderer` does. If it does not, scenes
   are still observed but the camera is not, so orbit falls back to pixels.
   No r3f-stack project has been built or checked through Docker yet.

`check.py --project DIR --out DIR` checks one project directory.

The smoke tests (2026-09-26) ran the two fixtures through Docker; no model
was involved.

- `sample_three.html`: every known answer came back. That was 401 voxels
  (400 instances plus 1 mesh), 4 instance colours, orbit detected by the
  camera, and only `cdn.jsdelivr.net:443` through the gate.
- `sample_canvas2d.html`: three.js was not seen. The check recorded 1 page
  error, `example.com` blocked, non-blank, and orbit false by pixels.

## Files

`bench/voxel/results/<tag>/`:

| file | tracked | what |
|---|---|---|
| `manifest.json` | yes | the task, the prompt, its sha256, and each arm as sent (r3f-stack: also the stack sentence and the sentence it replaces) |
| `summary.jsonl` | yes | one row per attempt, append-only: the task, the sha256 of the prompt sent, status, HTTP status, finish, usage, `x_yamadori.usage`, wall time (429 waits excluded), the stack's decisions (tier, route, fan-out, deep thinking, skills, fold-backs, images, budget, cache, energy) and the extraction |
| `checks.jsonl` | yes | one row per check: the task and the measures above (r3f-stack: also `build`, `stack` and `page_checked`) |
| `reps/<arm>__r<n>__a<k>/` | no | `response.json` (the whole body: content, reasoning, the full `x_yamadori`), `page.html` or `project/` (r3f-stack), `extract.json`, `build/` (r3f-stack: the step logs, exit codes, `dist/`), `check/` (`browser.json`, `check.json`, the shots) |
| `index.html`, `index_blind.html`, `blind/`, `blind_key.json` | no | the compare pages. The blind page hides arm names, tokens, wall time and the stack record, and copies the shots and pages under `blind/` so no path names an arm. The key is a separate file. |

## Tests

`python bench/voxel/test_voxel.py` runs offline with fake responses and a
tiny page. It covers:

- the verbatim prompt and the default arm
- the body and headers as sent
- extraction: fenced, raw, cut and multi-block answers
- each status, with a 429 recorded as `not_run`
- resume
- the preflight refusals, with nothing sent
- the key never printed or written
- the BUSY list naming this runner
- the route decision
- the JS parsing under node, and the stats function on a fake scene with
  known answers
- the pixel measures, the check summary, and the compare page (blind
  included)
- r3f-stack:
  - the prompt pin
  - the prompt is `PROMPT` with only its last sentence replaced by the
    operator's (the same string as the Octopus V4 prompt's), and nothing
    of ours (no pins, layout, rules or convention)
  - single-html is unchanged and still the default
  - extraction on hand-written answers: convention, info-string, heading
    and comment paths; nested README fences; unclosed blocks; refused
    absolute, `..` and drive paths; duplicates; a project under its own
    folder; an unnamed HTML page taken as `index.html`
  - the rep record and the manifest
  - the task-mismatch and changed-prompt refusals
  - the static stack checks on a small hand-written project on the named
    stack, and on one with r3f v9, no koota and an unused math import,
    where every check fails; the installed version winning over
    package.json
  - `check_project` with a fake builder and a fake browser: a failed build
    is a measurement, and a build that could not run is an error
  - `serve_app.sh` passes `bash -n`, its serving path is unchanged, and it
    has the `npm` and `js build` modes

No Docker build of an r3f-stack project has been run.

`scripts/run_tests.py` runs it.
