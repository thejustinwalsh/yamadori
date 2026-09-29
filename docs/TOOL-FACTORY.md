# The tool factory: a new MCP server, from a source to a probed `yama_*` tool

PLAN, 2026-09-29. **Nothing here is built.** No code changed, nothing was sent
to `:1234` or `:11434`, no container started and no GPU ran for this
document. Every "exists" below was read in the code on 2026-09-29. One offline
check was run to write it: `skill_limits.doubt` and `json.dumps` over
`mcp_config.PACKAGELENS` (section 5, results quoted there).

Operator, 2026-09-29: "Being able to ingest new server MCP tools and have a
prompt that extracts the information, lets you craft a test run, and verifies
it all works is the most useful thing we could build right now, this would be
its own tools page. The goal is to allow what we are doing right now to be
exposed for a user to setup without us maintaining it, durable pipeline,
exact prompts and verifications to get the exact results that work, all done
by the local model. ... make this plan based on our evidence."

Every design choice carries its class, the repo's rule (AGENTS.md "Claims
carry their evidence"; memory "no invented numbers"):

| tag | meaning |
|---|---|
| **[operator]** | the operator's decision, quoted with its date |
| **[derived]** | derived from a real constraint (security, correctness, a protocol's, engine's or harness's limit), the derivation beside it |
| **[measured]** | measured: the script or log and its n |
| **[open]** | not decided; listed in section 12 for the operator |

No threshold, count or cap in this plan is ours. Where a pipeline needs one
(how many trials, what call rate arms a tool), it is the USER's, entered on
the page, recorded with the author, and the page says what that n can show
(PROTOCOL rule 4).

---

## 0. What we did by hand, and what went wrong elsewhere

"What we are doing right now" is the PackageLens path of 2026-09-28/29. The
factory is that path, made durable and run by the local model where a model
step exists. Each row is the evidence a later section cites.

| step | what was done | where it is recorded |
|---|---|---|
| E1 choose + approve | npm `packagelens-mcp@0.1.11` (MIT); operator-approved download, 2026-09-28, "Yes get package lens" | `mcp/mcp_config.py` header; `models/manifest.yaml` `mcp-packagelens` |
| E2 pin the tree | 16 packages, every sha512 in `package-lock.json`; `npm ci --omit=dev --ignore-scripts`; `npm audit signatures` at build (16 verified registry signatures, 2 attestations); the version asserted in the image | `mcp_servers/packagelens/Dockerfile`, `package-lock.json`; manifest `tree` |
| E3 source audit | `dist/` of the server and the SDK read BEFORE it ran: only `fetch()` GETs (12 s abort, one retry) to 8 named registry hosts; executes nothing (`child_process` only in the SDK's CLIENT transport, never imported by the server); writes nothing; reads its own `package.json` and stats manifests in its cwd; sends `GITHUB_TOKEN` only if set (it is not); its tool descriptions carry "CRITICAL RULES" prose aimed at the model | manifest `mcp-packagelens` comment block |
| E4 contain | long-lived stdio container: pinned image (id checked before every start), own `--internal` network whose one way out is the egress gate, uid 1000, `--cap-drop ALL`, no-new-privileges, read-only root, no credential | `mcp/mcp_host.py` `run_argv`, `check_image`; `bench/sandbox/sandbox_net.py`, `egress_gate.py` |
| E5 choose tools | 3 of its 8; each left-out tool has a reason read from the source (`smart_package_info` reads a field a packument lacks; `smart_get_dependencies` always empty without a version; the usage snippet duplicates the README; downloads/compare are popularity, not discovery) | `mcp/mcp_config.py` "THE SUBSET" |
| E6 replace descriptions | vendor text never passed; ours open with the question ("Answers '...'"), say when to call, end with the server-tool sentence; `ecosystem` made REQUIRED (without it the server guesses, or downloads PyPI's whole index); one positive system line | `mcp_config.PACKAGELENS`; AGENTS.md THE TOOL RECIPE |
| E7 derive the timeout | 48.6 s from the server's own `dist/http.js` (2 sequential GETs x (2 x 12 s + 0.3 s)) | `mcp_config.PL_CALL_TIMEOUT_S`, `call_timeout_why` |
| E8 smoke | `python mcp/mcp_host.py smoke packagelens`: real server, 9 fixed calls through render + screen, no model; the expected answers ("pmndrs math" -> `math`, "koota" -> `koota`, r3f versions include the 10.0.0 alphas) were judged BY EYE -- `SMOKE_CALLS` carries inputs only and the run wrote no file | `mcp_host.SMOKE_CALLS`, `cmd_smoke` |
| E9 probe | Pi 0.87.1 in the harness box, tier medium, `{"mcp_tools": true, "skills": false, "retrieval": false, "check_code": false}` (a clean context), the vague pagoda prompt, stopped at the first install command, 10 requests / 15 min bounds (the coordinator's): **called a lookup before installing in 4/4 valid trials**; trial 5 `not_run` (a slot busy at preflight) | `bench/mcp/lookup_probe.py`; `bench/mcp/results/lookup_probe.jsonl` (5 rows; n=4 valid, one prompt, no ablation); relay logs `C:\Users\jwals\octo\logs\probe-lookup-pi-medium-{1..4}` |
| E10 what it did with the results | trial 3 guessed `@react-three/koota` and `@pmndrs/math` (both `NOT_FOUND`), then searched and found `koota` and `math`; its manifest pairs `@react-three/drei ^10.7.9` with `@react-three/fiber 10.0.0-canary.33238a1` -- version sets that do not install together, because nothing returned peer dependencies | `lookup_probe.jsonl` trial 3 `calls_before_install`, `manifests` |
| E11 fix the tool, re-smoke | peers added from the npm abbreviated packument (`npm_peers`); an empty registry README read from the package's GitHub repository (`github_readme`); re-smoked. **No probe ran after the change** (the jsonl has no later row) | `mcp/mcp_host.py` docstring (2026-09-29) |
| E12 lock the recipe | operator, 2026-09-29, "lock it in": seven rules; "none is offered by default until its probe passes" | AGENTS.md "Tool descriptions are prompts" -> THE TOOL RECIPE; memory `tool-recipe` |

What the same evidence says went wrong, and the factory must not repeat:

- **The probe's own "use" metric was broken in 4/4 rows.** `install_names`
  is `create-vite` (trials 1-2, a scaffold) or `2>&1` (trials 3-4: the
  regex split of `npm install 2>&1 | tail -25` kept the redirection as a
  package name, so the manifest fallback never ran); `install_names_found`
  is empty in every row. PROTOCOL rule 3 ("exactly 0 or 100 percent is a
  harness bug until proven otherwise") and rule 8 ("parse the language, do
  not pattern-match it") both apply. The call-rate half is sound: every
  trial records the calls with their arguments and returned names.
- **The probe ran as client traffic.** Its key is `pi-dogfood.key`
  (`bench/octopus/run.py` `PI_KEY_FILE`); `corpus.account_traffic` calls an
  account "test" only when its label starts with `live-test` or
  `claude-dogfood` (`TEST_ACCOUNT_LABELS`), so these requests are rows the
  learners may read.
- **Unproven content armed automatically hurt the model.** Pagoda p1
  (2026-09-28, xhigh, our layer on): per-step skills injected ~3k characters
  of off-topic crafts every step; h4-h6 skills and plans carried doubt the
  model acted on (memory `pagoda-evidence-2026-09-28`, `no-doubt-in-
  injections`). The skills pipeline gained REVIEW and PROVE on 2026-09-28
  for this reason (AGENTS.md "Skills").
- **An abstract tool was never called.** `yama_think_deeply`, offered for
  whole runs, was called 0 times voluntarily; three "Remember (server tool
  ...)" recall lines on pagoda-h5 did not change that (AGENTS.md
  "Heavy-handed, at the right time"; THE TOOL RECIPE's evidence line).
- **The model does not know package facts.** It guessed `@pmndrs/maths`
  and `@react-three-fiber` (pagoda p2), `@pmndrs/math`,
  `@react-three/koota` (probe trial 3) and `koot` (trial 4). A pipeline
  that lets the model WRITE an expected answer would encode the same
  guesses (section 3.9).

---

## 1. Principles

1. **One more staged pipeline on the existing machine** -- a dataset row of
   a new kind whose stages are jobs on `mcp/jobs.py`'s queue, advanced by
   `mcp/worker.py`. No new orchestrator. [operator, 2026-09-27, for
   onboarding: "That is a stateful pipeline in action, we already have the
   durable machine too."; 2026-09-29: "durable pipeline"]
2. **The model proposes, code verifies.** Every model stage is a versioned
   template pinned by a test; every reply is parsed strictly and checked
   deterministically; a refused proposal is recorded with why. [operator
   rule of the skills pipeline, `mcp/skill_prompts.py`; research: SkillsBench
   2602.12670 self-generated -8.1 to -11.5 pp unless verified, ASI
   verification +4.2 (SKILLS-RESEARCH.md takeaway 2)]
3. **The model never names a fact.** Package names, versions, licences,
   hosts, expected answers and pass/fail come from the registry, the source,
   the server's own output, code checks or the user -- never from model
   memory. [measured: section 0's wrong guesses; the onboarding rule "Nothing
   in onboarding asks a model to name a package ... a version, a licence"
   (PACKAGE-ONBOARDING.md 1.3)]
4. **Nothing arms unproven.** A tool reaches main only after its smoke
   passed the user's confirmed expectations AND its probe passed the user's
   gate on this model. [operator, 2026-09-29: THE TOOL RECIPE rule 7 and
   "none is offered by default until its probe passes"]
5. **No review gate; only the holds the rules require.** The pipeline runs
   unattended [operator, 2026-09-22, memory `dataset-pipeline-no-review-
   gate`]. It holds for a person only where a rule makes a person the only
   valid source: a download (approval), a credential (only the user can
   supply one), a licence with no verbatim quote (onboarding's clarify), the
   smoke expectations (principle 3: truth the model does not have), and the
   gate numbers (no invented numbers). Everything else is an optional edit.
6. **Untrusted code never runs on the host, and runs only after its audit.**
   The server runs in its own gated container; its tree is built with
   lifecycle scripts off. [derived: SELF-IMPROVEMENT-LOG #48/#49 -- a
   container on the default bridge reaches every host loopback service
   (docs/HARNESS-SANDBOX.md)]
7. **Test the real thing through the real door.** Smoke calls the real
   server; the probe drives a real harness through `:1234`. [AGENTS.md
   "Before you claim anything works" 3: "Never test around the stack";
   operator 2026-09-24: "If we don't have tests that exercise the real
   models, we don't have tests"]
8. **Server tools are about the work, not the workspace.** A tool that reads
   or writes the project, runs commands or drives a browser belongs in the
   harness's config. [derived: THE TOOL RECIPE rule 2 "does not touch your
   workspace"; AGENTS.md "apply_edit was cut because the harness owns
   writing"; memory `harness-tools-direction`, `no-harness-patches`]

---

## 2. The shape: a dataset of kind `mcp_server`

One dataset row per `server@version` (`datasets.KINDS` gains `mcp_server`,
beside `recipes`, `laya`, `package`). Stages:

```
submitted -> resolve -> [approve] -> tree -> audit -> audit_summary -> build
  -> list -> explore -> draft -> smoke_cases -> smoke_record -> [confirm]
  -> smoke_verify -> probe -> decide -> arm -> serving
                                          (watch: a new version opens a new row)
```

`[...]` is a HOLD for a person (principle 5), implemented like onboarding's
`clarify`: `datasets.missing()` lists what is needed, `onboarding.
held_for_person`'s idiom holds the stage, the page shows the form.

| stage | lane | model? | what it does | holds on |
|---|---|---|---|---|
| resolve | net | no | source -> kind, name, exact version, artefact URL, integrity, licence quotes, repository + commit (3.1) | its job |
| approve | -- | no | the download list with sources and sizes (3.2) | the user's approval |
| tree | net | no | the dependency tree pinned: lockfile, every integrity, signatures (3.3) | its job |
| audit | cpu | no | tree-sitter findings over the reachable source: network, exec, fs, env, native, dynamic (3.4) | its job; a HARD REFUSAL ends the row |
| audit_summary | gpu, idle | yes | the model summarises the findings with file:line quotes; code checks every quote (3.4) | its job |
| build | net | no | `docker build --pull=false` from the recipe; the image id recorded (3.5) | its job |
| list | net | no | start sandboxed under its own tag, `initialize`, `tools/list`; the gate's log (3.6) | its job; a credential it needs |
| explore | gpu, idle, then net | yes | the model proposes calls per upstream tool; code validates and runs them (3.7) | its job |
| draft | gpu, idle | yes | keep/drop, name, parts of the description, our schema, args mapping, overlaps, the system line; code assembles and lints (3.8) | its job |
| smoke_cases | gpu, idle | yes | the model proposes smoke inputs and probe prompts (3.9) | its job |
| smoke_record | net | no | runs the smoke inputs, records each result (3.9) | its job |
| confirm | -- | no | the user confirms smoke expectations, probe tasks, run settings, the gate (3.10) | the user |
| smoke_verify | net | no | runs the smoke again, checks every confirmed expectation (3.11) | its job |
| probe | gpu, idle | the SERVED model, as a client | trials through a real harness on `:1234` (3.12) | its job |
| decide | cpu | no | the user's gate over the valid trials (3.13) | its job |
| arm | cpu | no | writes the server's row to `servers.json` (3.13) | its job |

Lanes are the queue's existing `{gpu: 1, cpu: 4, net: 4}` (`jobs.LANES`).
Every gpu stage carries `"idle": true` so `worker.run_one` defers it through
`mcp/idle.py` without spending an attempt (built 2026-09-27 for onboarding).
[derived: AGENTS.md "One GPU consumer at a time"; operator, 2026-09-27,
decision 5 of PACKAGE-ONBOARDING.md for the idle gate; `IDLE_MINUTES` = 15
is "operator, 2026-09-27: keep ours"]

Idempotency and resume follow PACKAGE-ONBOARDING.md 1.2: each stage is a
function of recorded inputs, writes its outputs atomically, and a re-run of a
completed stage with the same input hashes is a no-op. `resolve` pins the
version on the first run; a newer `latest` never moves a running row (the
onboarding rule).

---

## 3. The stages

### 3.1 resolve

| source the user gives | resolves to | exists today |
|---|---|---|
| npm name, `name@version`, an npmjs.com URL | packument -> exact version (the link's, a stated major's newest, else `dist-tags.latest`: `package_resolve.choose_version`), `dist.tarball`, `dist.integrity`, `dist.unpackedSize`, `bin`, `repository` + `gitHead` | `package_resolve._npm`, `package_net.npm_packument` |
| PyPI name / URL | the version's files (wheel preferred: 3.3), sha256 digests, `project_urls` | `package_resolve._pypi` |
| GitHub URL | the repo at a ref -> commit; its manifest's package name; the REGISTRY artefact when that name@version is published there (it is what gets installed) | `package_resolve._github`, `commit_of_ref`, `tag_commit` |
| Docker image reference | the digest (`name@sha256:...`) -- a tag alone is refused until pinned; config (entrypoint, user, env, labels) read with `docker manifest inspect` / the registry API, no pull | new |
| remote MCP URL (streamable HTTP) | host, TLS, `initialize` answer's `serverInfo`; nothing to audit | new (section 10, M7) |

THE LICENCE, verbatim: `package_resolve.licence_quotes` (the LICENSE file at
the commit, the manifest's own `license` line, the tarball's LICENSE after
`deps.fetch_verified` extracts it). No quote holds the row for the user's
statement, recorded with provenance `operator` (`POST
/dash/api/dataset/answer`, exists); a restricted licence is surfaced by
`datasets.warnings()` (exists). [operator, 2026-09-27, onboarding; derived:
running and redistributing the server is a use its licence governs]

The user also gives a GOAL -- one line, what they want the tool for ("find a
package, its versions and how it is used before installing it"). It is the
only free text the model stages read about intent. [derived: the draft needs
the task the tools serve; PackageLens's goal was the operator's, memory
`pagoda-evidence-2026-09-28`: "if I type use pmndrs math I want to help it
find the package"]

### 3.2 approve (a hold)

The page lists every file the build will download, before any is: the
artefact and every package of the resolved tree (name, version, source URL,
`dist.unpackedSize` from its packument), the total, and a base image when it
is not in the local store (its compressed size from the registry manifest, as
`harness_box.BASE_PULL_MB` = 410 was recorded). APPROVE records the account,
the time and the exact list's sha256; a later change of the list (a re-
resolve) needs a new approval. [operator: E1's "Yes get package lens";
`mcp_host.cmd_build`: "Pulling it is a download for the operator to decide
on"; the stack's safety rules: a download needs explicit approval with
filename, source and size]

The tree's METADATA (packuments) is fetched before approval -- the same GETs
`resolve` makes; no tarball, no image layer. [derived: it is what makes the
list above possible]

### 3.3 tree

| runtime | how the tree is pinned | exists |
|---|---|---|
| Node | `npm install --package-lock-only --ignore-scripts` inside a throwaway container on the harness box's base image (local, pinned), behind the egress gate: packuments only, no tarball; then the build's `npm ci --omit=dev --ignore-scripts` fetches exactly the lockfile, every tarball checked against its sha512, and `npm audit signatures --omit=dev` fails the build on a bad signature | the PackageLens Dockerfile does the second half; the first half is new |
| Python | a hash-locked requirements file (`--require-hashes`) over WHEELS only; an sdist-only dependency runs its build backend at install, which is executing unaudited code, so it is refused or held (section 12) | new |
| Docker image | the digest is the pin; layer history recorded | new |

Recorded: the lockfile's sha256, every package's integrity and licence
(`license` field; a missing or restricted one surfaced), install scripts
declared anywhere in the tree (they stay off; the list is shown), and the
signature/attestation verdicts. PyPI attestations (PEP 740) are verified
where a project publishes them -- [open]: not researched for this plan.

### 3.4 audit, then audit_summary

AUDIT is code only (PROTOCOL rule 8: a parser, not regexes). The hand audit
(E3) answered five questions; each becomes a finding kind:

| question (E3) | finding | how code finds it |
|---|---|---|
| what does it fetch, from where, with which method? | `network` {file:line, API, literal host or URL, method when the call site states it, else "unknown"} | tree-sitter JS/TS/Python: calls of `fetch`, `http(s).request`, `undici`, `axios`, `net.connect`, `WebSocket`; `requests`, `httpx`, `urllib`, `aiohttp`, `socket`; string literals with a scheme or a registered hostname |
| does it execute anything? | `exec` | `child_process`, `execa`, `spawn`; `subprocess`, `os.system`, `pty`; `eval`, `new Function`, `vm`; `exec`/`compile` |
| does it write anything? | `fs_write` | `fs.write*`, `createWriteStream`, `mkdir`, `rm`; `open(..., "w"/"a")`, `pathlib` writes, `shutil` |
| what does it read from the environment? | `env` {name} | `process.env.X`, `os.environ[...]`, `getenv` |
| is anything unauditable? | `native`, `dynamic`, `minified` | `.node`/`.so`/`.dll`/`.wasm` files; `require`/`import()` with a non-literal argument; `deps.is_minified` |

SCOPE is what the server can run: the import graph from its entry point
(`bin` / `main` / the image entrypoint), resolved inside the pinned tree.
A finding in a file the entry cannot reach is listed as unreachable -- E3's
"`child_process` appears only in the SDK's client transport, which the server
never imports" is exactly this distinction. A `dynamic` finding makes the
graph unsound from that file on; everything the dynamic site could load is
treated as reachable. [derived: correctness of the reachability claim]

HARD REFUSALS (the row ends; the page says why and what would change it):

| refusal | derived from |
|---|---|
| the server is a WORKSPACE tool: its tools take a path, a command, a URL to drive, or carry `destructiveHint` / `readOnlyHint: false` (read at `list`), or reachable `fs_write` outside its temp dir | principle 8 |
| a reachable `native` binary or a `.wasm` module with no source in the tree | the audit cannot read it (principle 6) |
| an artefact that cannot be pinned (an image tag with no digest, a git branch with no commit) | memory `reproducible-everything`; `mcp_host.check_image` already refuses an unrecorded image |

Whether a reachable `exec` is a refusal or a finding the user acknowledges is
[open]: E3 found none, so there is no evidence on it.

THE EGRESS ALLOWLIST. The `network` findings' hosts are the server's
declared destinations. Today the gate allows ANY global address on 80/443
(`egress_gate.ALLOWED_PORTS`, "a CHOICE (2026-09-26)"), so a server can send
the model's arguments -- which may carry the user's code -- anywhere. The
factory adds a per-server host allowlist to the gate (the gate already
resolves every name itself and logs one line per decision), filled from the
audit's hosts plus hosts the user adds on the page; every `DENY` during
`list`, `explore`, smoke or the probe is a finding ("a host the audit
missed"). [derived: the arguments are the user's data; security]. For
PackageLens the audit's list is registry.npmjs.org, api.npmjs.org,
api.github.com, crates.io, pypi.org, rubygems.org, packagist.org, hex.pm
(E3).

CREDENTIALS. `env` findings that name a credential (the audit's list, plus
`skill_screen.check_credentials`'s shapes) are shown as "this server can use
X". None is set unless the user supplies it FOR THIS SERVER; its value never
enters `servers.json`, a record, an argv or a log -- passed as `-e NAME` from
the starting process's environment, as `harness_box.py` passes the harness key
[derived: the stack's credential rule; PackageLens runs with no
`GITHUB_TOKEN`, E4]. Where the store lives is [open].

AUDIT_SUMMARY is the model's first job: from the findings (never the raw
source), a short plain-language account per question of E3, each sentence
citing a finding id; a sentence whose quote is not verbatim at its
file:line is dropped (`worker.find_verbatim`, exists). The source files and
the vendor's own texts enter the prompt only inside the `DATA_PARAGRAPH`
frame (`skill_prompts.DATA_PARAGRAPH`; docs/INJECTION.md F1: 1/310 vs 13/350
compliance with the hardened paragraph, measured on answering, not on these
prompts). The summary DECIDES nothing: refusals are the code's.

### 3.5 build

`docker build --pull=false` from a recipe the factory writes: the PackageLens
Dockerfile generalised -- the pinned base (already local, or approved at
3.2), the lockfile, scripts off, signatures checked, the version asserted,
`USER` non-root, an empty working directory, the entrypoint. The image id is
recorded where `mcp_host.check_image` reads it, so a rebuild with a new id
refuses to start until recorded (the existing rule). Where the factory's
recipes and records live -- `mcp_servers/<id>/` + `models/manifest.yaml` (the
repo, as PackageLens) or the state directory + a factory manifest that
`scripts/verify_artifacts.py` also reads -- is [open]: a user's pipeline that
writes into the repo tree is a new thing.

`mcp_host.run_argv` gains the resource limits `harness_box` already applies
(pids, memory, cpu): the host starts containers with none today. The values
are [open]: harness_box's are the only precedent.

### 3.6 list

The real server, started exactly as the proxy will start it but under its
own tag (the `cmd_smoke` pattern: `Server(spec, tag=...)`, its own gated
network, torn down after): `initialize`, `tools/list`. Recorded: the
server's `serverInfo` and protocol revision, every upstream tool (name,
description, `inputSchema`, `outputSchema`, annotations) and a sha256 of the
list, the size of each as an OpenAI function definition
(`bench/sandbox/mcp_probe.mjs` already measures `chars`), and the gate log.
A vendor description is DATA: shown escaped and labelled on the page, read by
the model only inside the data frame, never sent to main. [E3/E6: the
PackageLens descriptions carried "CRITICAL RULES" aimed at the model]

KNOWN LIMIT: `mcp_host.StdioClient` speaks protocol `2024-11-05`, PackageLens's
SDK's revision, and reads only `text` content (`_text_of`). A server that
needs a later revision's features or answers with `structuredContent`,
images or resources needs the client extended (section 10, B3).

### 3.7 explore

What the hand path learned from the source and from calling the server --
that a tool returns nothing without a version, that the server guesses an
ecosystem when not told -- the factory learns by calling it. The model
proposes calls per upstream tool from the goal, the schema and the audit
summary; code keeps a proposal only if its arguments validate against the
upstream `inputSchema`, then runs it through the sandboxed server and records
the raw result, its bytes, the time and the screen's verdict
(`skill_screen.screen_fetched`, exists). One proposal kind is required per
optional parameter: the same call WITH and WITHOUT it, so the draft can
evidence "this argument removes ambiguity" (E6's `ecosystem`). How many calls
the model may propose per tool is [open] (each is a request to a third
party; the skills PROVE stage's operator figure is "2-3" probes, a precedent,
not this number).

### 3.8 draft

The model PROPOSES PARTS; code ASSEMBLES the description, so the recipe's
structure holds by construction, and lints every part (section 5).

Per upstream tool, one proposal:

| field | the model proposes | code verifies |
|---|---|---|
| verdict | keep / drop | a drop carries one reason from a closed set, each with an evidence reference that must exist: `overlaps_kept:<tool>` (a kept tool), `overlaps_client:<names>` (names in a harness fixture, below), `not_task_advancing` (the goal), `broken:<explore id>` (that call failed or returned nothing), `side_effect` / `workspace` (an annotation or an audit finding) |
| question | the one question it answers | one sentence ending in `?`; becomes `Answers '<question>'` |
| returns | what the result holds | any field it names in backticks exists in an explore result |
| when | the moments in the model's own work to call it | at least one clause; becomes `Call it <when>.` |
| name | `yama_<words>` | section 5, N1-N4 |
| parameters | our schema: a subset or renaming of the upstream's, descriptions | valid JSON Schema; every upstream `required` parameter is mapped or fixed; enum values a subset of the upstream's |
| required | which of ours are required | an optional upstream parameter becomes required only with a with/without explore pair whose results differ (the E6 evidence, made mechanical) |
| args | {ours: upstream} | every key in our schema; every value in the upstream schema |
| overlaps | client tool names answering the same question | each name occurs in a harness fixture's tool list (`bench/harness_shapes/*/`) -- the list `proxy.tool_conflicts` withholds beside |
| render | a field order for a JSON result (paths) | every path exists in an explore result; else the generic render (below) |

Per server: the system LINE, `To <goal clause>, call {tools} before you
<moment clause>.` -- the moment clause is the probe's moment (3.9), so the
line and the measurement name the same event.

The assembled description is
`Answers '<question>': <returns>. Call it <when>. ` + `mcp_config._SERVER_TOOL`
("A server tool: it runs on the Yamadori server and does not touch your
workspace."), the PackageLens shape. [operator, 2026-09-29: THE TOOL RECIPE
rules 1-4]

RENDER. PackageLens has three hand-written renderers (`mcp_host.RENDER`:
search, versions, readme) and two hand-written augmentations (`npm_peers`,
`github_readme`). A user's server gets a GENERIC render: text content as the
server wrote it; a JSON result flattened deterministically in the server's
own order (or the verified field order above), under `SOURCE:` and the data
note, screened, capped at main's tool-result cap like every result. What the
factory cannot do is WRITE CODE: an augmentation like peers is a code change
by us, or another server's tool. The probe's use checks (3.9) are how such a
gap is FOUND (E10 found peers); the page reports it with the trial evidence
and does not pretend to fix it. [derived: the model does not write code
that runs in the proxy's request path]

### 3.9 smoke_cases and smoke_record: crafting the tests

**Smoke.** The model proposes INPUTS only: per kept tool, a call the goal
makes typical, a not-found call, and each with/without ambiguity pair. Code
validates them against OUR schema and `smoke_record` runs them through
`mcp_host.run_tool` (render + screen + failure envelope: what main would
read). Then the model proposes candidate ASSERTIONS read off the recorded
result, from a closed set checked by code:

| assertion | passes when |
|---|---|
| `ok` / `error:<CODE>` | the envelope's `ok`, or its `error` code (`NOT_FOUND`, `BAD_ARGUMENTS`, ...) |
| `first:<text>` | the first result line contains it ("best answer first", recipe rule 6) |
| `contains:<text>` / `lacks:<text>` | the text does / does not occur |
| `matches:<regex>` | the regex compiles, does not match the empty text, and matches |

The user confirms, edits or adds assertions (3.10). The model's candidates
are proposals to tick, never truth: its own guesses of these facts were wrong
(section 0). [measured: pagoda p2, probe trials 3-4] PackageLens's E8
expectations, written as such cases, are the fixture every test of this stage
starts from.

**Probe tasks.** The model proposes, from the goal and the kept tools'
questions:

- SHOULD-CALL prompts: realistic work in which the moment arrives and the
  tool is for it, naming the thing loosely, as the pagoda prompt named
  "pmndrs math" and never `math` (E9).
- SHOULD-NOT prompts: realistic work in which the tool is not for it (the
  thing is named exactly, or the work does not reach the moment). E9 had
  none; over-calling is a documented failure (SRA 2604.24594: load rates
  insensitive to need; SMART 2502.11435 [A]: over-use), so the factory
  measures it. [research; not yet evidenced on Bonsai]

Code refuses a prompt that names a kept tool, contains `yama_`, contains a
confirmed smoke answer (it would hand over the result), or shares an 8-word
run with a description (the leak rule `skill_prove` and
`test_skill_factory` already apply). The user edits freely (3.10).

The MOMENT and the USE CHECKS are LABELS from code tables, like `run_check`'s
`VERIFY_CHECKS` ("never a command line", AGENTS.md "Checks"); adding one is a
code change with a test.

| moment | detected from the harness's events | evidence |
|---|---|---|
| `install` | the first bash command matching `lookup_probe.INSTALL_RX` | measured: E9, n=4 |
| `first_write` | the first write/edit tool call | new |
| `first_import` | the first written file whose imports (tree-sitter) name a package | new |
| `answer` | the harness's first final answer | new |

| use check | passes when | evidence |
|---|---|---|
| `called_before_moment` | a call of a kept tool precedes the moment | E9 |
| `arguments_valid` | no `BAD_ARGUMENTS` among the calls | new |
| `results_used` | a name our results returned appears in the moment's action or a manifest written by then, the command parsed with tree-sitter bash (`skill_screen.parse_commands`) | E9's `install_names_found`, with its parser defect fixed (section 0) |
| `project_installs` | after the stop, `npm install --dry-run --ignore-scripts` in the probe's project, inside the harness box, exits 0 | derived: E10 is exactly this failure; it would have caught it |

### 3.10 confirm (a hold)

One form, prefilled with the model's proposals:

- the smoke table: each input, the recorded result (screened, escaped), the
  candidate assertions to tick or edit, fields to add one;
- the probe tasks: each prompt editable, its kind, its moment and use checks;
- the run settings: harness (Pi, the one measured; others with M8), tier,
  trials per task, max requests and max minutes per trial -- all EMPTY until
  the user fills them ("no default: a bound nobody chose is not written
  here", `lookup_probe.py`), with E9's values shown as what was used before
  (medium; 10 requests; 15 minutes);
- the GATE: for each should-call task, calls at the moment in at least ___ of
  the valid trials; for each should-not task, at most ___; each use check in
  at least ___. Empty until the user fills it. Beside it, what the entered n
  can show: with n trials, one miss moves the rate by 1/n; E9 is 4 of 4 on one
  prompt (PROTOCOL rule 4; rule 10: "one experiment is a data point").

CONFIRM records the account, the time and the sha256 of every confirmed
object. Any later edit re-opens `smoke_verify` and `probe` for that version.

### 3.11 smoke_verify

The confirmed cases, run again against the real sandboxed server; every
assertion pass/fail with the result it read; the gate log's `DENY` lines.
All pass -> `probe`. A failure holds the row with the failing case shown;
nothing is retried silently (PROTOCOL rule 2). [E8, made mechanical: the
smoke's answers were judged by eye]

### 3.12 probe

`bench/mcp/lookup_probe.py` generalised into product code
(`mcp/tool_probe.py`), one trial per job row so a killed worker resumes at
the next trial:

1. PREFLIGHT (`run.pi_preflight`'s checks, through `mcp/idle.py`): no other
   GPU consumer, every slot idle, the proxy answers, Docker answers, the
   harness box image, the probe key. Not idle -> `jobs.defer`; a 429 is NOT
   RUN, never a failure (AGENTS.md "Before you claim anything works" 4).
2. THE CANDIDATE, not yet armed: the proxy offers it to the probe's
   conversations only -- a header naming the candidate row
   (`X-Yamadori-Features {"mcp_candidate": "<id>@<version>"}`), honoured for
   a TEST account only, the pattern `idle_clear_s` uses ("a test account may
   override ... refused and recorded otherwise"). [derived: principle 4 --
   the tool cannot be offered to users to measure it]
3. A CLEAN CONTEXT: the features header E9 used, `{"mcp_tools": true,
   "skills": false, "retrieval": false, "check_code": false}`, at the tier
   the user chose. [operator: THE TOOL RECIPE rule 5] At `xhigh`/`max` the
   kickoff plan and `yama_think_deeply` are ours too, so a clean context
   there needs them off by header as well; E9 ran at `medium` only.
4. THE HARNESS in the harness box (`harness_box.py run`, the default
   loadout), a fresh project folder, every request through the recording
   relay so each response's `x_yamadori.mcp` is on record.
5. THE STOP at the moment, the request bound or the time bound; the
   container removed by name (E9's `Watch`).
6. THE ROW, per trial, E9's schema plus the task id and kind, the moment,
   each use check's verdict with its evidence, and the gate log's `DENY`s.
   `invalid` when the header did not reach the proxy, the candidate's tools
   were not on main, the tier differs, or a request was answered 429
   (E9's `summarise` checks, plus the 429).

The probe KEY is a test-labelled account's (`TEST_ACCOUNT_LABELS` gains the
probe's label, or the account uses `live-test`), so probe traffic is never
learned from. [derived: section 0, the `pi-dogfood` key is client traffic]
Where the worker reads it is [open] (the live suite takes `--key-file`).

This is the one stage that deliberately enters `:1234`, which internal
callers must not (AGENTS.md "One door to the model"): it is the live suite's
pattern -- a real client through the door users use -- not internal
generation. It is recorded as test traffic, and admission treats it as any
client.

MEASURED COST (E9 `wall_s`): 32.0, 60.0, 246.1 and 239.6 s per trial;
`requests_before_install` 1, 3, 1, 5 (n=4, Pi, medium). A task's trials run
back to back on the one gpu lane.

### 3.13 decide and arm

DECIDE applies the user's gate to the VALID trials (invalid and not-run rows
are listed, never counted) and writes the verdict with every number and its
n. Pass -> ARM; fail -> the row stays `not armed`, with the failing tasks'
trials linked. The page's next steps are the recipe's own order: rewrite the
description (AGENTS.md: "When a tool is mis-selected, fix its description
before touching the system prompt"), then the line, then drop a tool. There
is no automatic rewrite-and-retry loop: a rewrite is a new draft version and
a new probe. [derived: a loop that edits until the gate passes on the same
tasks is fitting the test (memory `no-graded-followups`)]

ARM writes the server's row to `index/mcp/servers.json` through
`mcp_config.save` (built, validated, not routed) with `enabled: true`, the
tiers the user chose, and `verified` {row id, version, image id, the verdict's
sha256}. `mcp_config.load` re-reads on the file's mtime and `mcp_host.get`
restarts a server whose spec changed, so no proxy restart is needed; the
tools reach NEW conversations only, since a conversation's offer is decided
on its first request and kept (`proxy._mcp_offer`: a tool list that changes
mid-conversation changes the cached system block).

ONE TRAP: `mcp_config.load` returns the built-in DEFAULT only while the file
does not exist. The first write must carry PackageLens's row, or PackageLens
disappears. The factory's first act is the PackageLens backfill (M0), which
writes it.

### 3.14 serving: operate

| control | what it does | exists |
|---|---|---|
| enable / disable | not offered to new conversations; a kept one gets `MCP_SERVER_OFF` | `mcp_config.set_enabled` (not routed) |
| tiers | per server, intersected with the `mcp_tools` switch (`medium` and up) | new field |
| watch | the registry's newest version, polled like the skills pipeline's watches (`skill_pipeline.schedule_watches`); a new version opens a new row through the whole pipeline while the old serves, and replaces it when it arms | new; the "serve the old until the new arms" rule is onboarding's decision 2 [operator, 2026-09-27, for packages; for tools [open]] |
| drift | at every start, the `tools/list` hash against the verified one; a kept tool whose schema changed is not offered until `explore` -> `smoke_verify` pass again | `_start` already logs missing upstream tools; the hash is new [derived: our args mapping is only correct for the schema it was checked on] |
| image | a rebuilt image refuses to start until recorded | `mcp_host.check_image` |
| per-call record | `x_yamadori.mcp.calls[]` {tool, server, upstream, ms, ok, bytes, error, screen, args (120 chars)} | exists on every response; NOT persisted -- the corpus keeps `tool_call` / `tool_result` rows by name. A `mcp_call` corpus event with the record's fields is new |
| usage panel | per tool: calls per day, error codes, p50 ms, screen strips, test vs client traffic | new, from the `mcp_call` events |
| results | rendered, screened by `skill_screen.screen_fetched`, framed as data, failures carry remedies | exists |

---

## 4. The prompts (`mcp/tool_prompts.py`)

The skill factory's rules, unchanged: each template is versioned, its version
recorded on everything it produced, its hash pinned by a test so an
unversioned edit fails (`mcp/test_skill_factory.py` does this for
`skill_prompts`); each model stage runs as a helper job `tool.<purpose>`
through `mcp/model.py` (the one door; thinking capped at `HELPER_THINKING`,
as `skill.<purpose>` jobs are); each carries `DATA_PARAGRAPH` around what it
reads; each reply is parsed strictly. Every wording is a CHOICE until
measured (AGENTS.md "Prompting this model": a decision table beats prose;
prohibitions sparingly).

| template | stage | reads | proposes | code decides |
|---|---|---|---|---|
| `audit_summary/1` | audit_summary | the findings table (never raw source) | per question, sentences citing finding ids with quotes | a quote not verbatim at its file:line drops the sentence |
| `explore/1` | explore | goal, upstream schemas, the audit summary, vendor descriptions (data) | calls, with/without pairs | schema validation; the run |
| `triage/1` | draft | goal, schemas, explore results (screened), vendor descriptions (data), the harness tool catalogue | keep/drop per tool with a reason kind and evidence id | the reason's evidence exists |
| `describe/1` | draft | the same, for kept tools | question, returns, when, name, parameters, required, args, overlaps, render order | section 5's lint; assembly |
| `line/1` | draft | goal, kept names, the moment label | goal clause, moment clause | section 5's L1-L4 |
| `smoke/1` | smoke_cases | kept tools, the goal | inputs; after the record, candidate assertions | schema validation; assertion kinds compile |
| `probe_tasks/1` | smoke_cases | goal, kept tools' questions, the moment table | should / should-not prompts, moment, use checks | the leak rules (3.9); labels exist |

The page shows, for every model stage of every row: the template id and
version, the exact rendered system and user text sent, the raw reply, the
parsed proposal and each code verdict (`GET /dash/api/skill-factory/prompts`
is the precedent for templates; the rendered text is new). [operator,
2026-09-29: "exact prompts and verifications"]

---

## 5. The lint: THE TOOL RECIPE as code

Each rule of THE TOOL RECIPE (operator, 2026-09-29) and the naming rules
(AGENTS.md "Naming") as a check; FAIL blocks the draft, REPORT is shown.

| id | check | from | kind |
|---|---|---|---|
| D1 | the description opens `Answers '` and the question ends `?'` | recipe 2 | FAIL (by construction) |
| D2 | a `Call it ...` clause follows | recipe 2 | FAIL (by construction) |
| D3 | it ends with `mcp_config._SERVER_TOOL` exactly | recipe 2 | FAIL (by construction) |
| D4 | it shares no 8-word run with the vendor's description | recipe 2 "replaced, never passed through" | FAIL |
| D5 | `skill_screen`: AI-directed text, hidden markup, credentials, invisible characters | the one screen | FAIL |
| D6 | `skill_limits.doubt` (the ASSURED VOICE) | memory `no-doubt-in-injections`: "anything we inject (plan, skill body, deep-thinking hand-off, directive, tool description) states the current way" | FAIL -- see the finding below |
| D7 | no all-caps directive words (CRITICAL, IMPORTANT, MUST) | recipe 4 "no CRITICAL"; E3's vendor "CRITICAL RULES" | FAIL |
| D8 | prohibitions counted | AGENTS.md "Prompting this model": at most two, each naming an observed failure (10.7 / 10.0 / 9.3, a measurement AT RISK, CONSTRAINTS #31) | REPORT |
| D9 | characters as an OpenAI function definition, per tool and for all of ours on main | "Less is More" 2411.15399 (fewer tools up to +71%); "Tool Calling is Linearly Readable" 2605.07990 (prompt length drives the collapse) | REPORT (no cap is ours) |
| L1 | the line is one sentence naming every offered tool through `{tools}` | recipe 4 | FAIL |
| L2 | no prohibition, no all-caps word | recipe 4 | FAIL |
| L3 | its moment clause is the probe's moment | derived: the line and the measurement name the same event | FAIL |
| L4 | `skill_limits.doubt` | as D6 | FAIL |
| N1 | `yama_` + snake_case | AGENTS.md: every tool of ours on main is `yama_*`; `mcp_config.validate` | FAIL |
| N2 | unique after `proxy.tool_conflicts`' normalisation (case, `-`/`_`, a plural s) against every `yama_*`, `proxy.LEGACY_TOOL_NAMES` and every harness fixture's tool | AGENTS.md, `tool_conflicts` | FAIL |
| N3 | verb first | AGENTS.md "Naming": "snake_case, verb first, spelled out" | see the finding below |
| N4 | `_opt` when the answer may be absent | AGENTS.md "Naming" | REPORT |
| A1 | every upstream `required` parameter is mapped or fixed | correctness | FAIL |
| A2 | a required-by-us optional parameter has its with/without evidence | recipe 3, E6 | FAIL |
| C1 | tools kept per server and ours on main in total | recipe 1 "few"; OpenAI [V] "fewer than 20 functions" at a time | REPORT (the user's call) |

PackageLens through it (run offline for this plan, `skill_limits.doubt` and
`json.dumps(mcp_config.definition(t))` over `mcp_config.PACKAGELENS`):

- `yama_find_package` 543 characters, `yama_package_versions` 478,
  `yama_package_readme` 500; as function definitions 3,378 characters for the
  three. The line passes `doubt`.
- **`yama_package_readme` FAILS D6**: `skill_limits.doubt` returns
  `instability` on "a library you have not used, or whose API may differ from
  what you remember". It was probed with that wording (E9: 4/4, README calls
  in every trial).
- **`yama_package_versions` and `yama_package_readme` fail N3** (noun first);
  they too were probed as they are.

Whether D6 and N3 bind existing, probed tools -- and whether N3 is FAIL or
REPORT -- is the operator's (section 12). The factory must not quietly
exempt them.

**Fixed 2026-09-29 (coordinator, relaying the operator's resolver request):**
`yama_package_versions` -> `yama_list_package_versions`,
`yama_package_readme` -> `yama_read_package_readme` (N3; the old names stay
readable: `mcp_config` `legacy`, so stored ledger hops and kept offers still
resolve), and the readme description's "or whose API may differ from what you
remember" removed (D6). `mcp/test_mcp_host.py`
`test_the_names_and_descriptions` gates `skill_limits.doubt` on every
description and the line, and the verb-first names. The renamed tools, and
the new `yama_resolve_packages`, have **not been re-probed**.

---

## 6. The data model

### 6.1 `index/mcp/servers.json` (`mcp_config`, version 1 -> 2)

What is SERVED. Version 2 adds, per server: `tiers` (list), `source` {kind,
locator, version, artefact, integrity}, `hosts` (the gate allowlist),
`credentials` [names only], `verified` {row, version, image_id, verdict
sha256, at}, and per tool `origin` {template versions, draft sha256} and
`render` `generic` (with an optional field order). `validate` gains the new
fields; a version-1 file still loads (PackageLens's hand row keeps
`render` search/versions/readme). No secret is ever in it (`dash_mcp`'s rule).

### 6.2 The run records: `index/mcp/factory/<row>/`

One directory per dataset row (`server@version`), each file written
atomically with the sha256 of its inputs:

| file | holds |
|---|---|
| `resolution.json` | the source, the rule that chose the version, integrity, licence quotes, repository + commit |
| `approval.json` | the download list, its sha256, who approved, when (or declined) |
| `tree.json` | lockfile sha256, packages {name, version, integrity, licence, install scripts declared}, signature verdicts |
| `audit.json` | findings {id, kind, file:line, excerpt, reachable}, the import graph's entry and unsound points, refusals |
| `audit_summary.json` | the template version, the rendered prompt's sha256, the reply, kept and dropped sentences with why |
| `image.json` | recipe files' sha256, base, image id |
| `list.json` | serverInfo, protocol, the raw tools/list, its sha256, chars per tool, gate log counts |
| `explore.jsonl` | one row per call: proposal, validation, args, result (screened), bytes, ms, error |
| `draft.json` | per tool: the proposal, each lint verdict, the assembled definition; the line; template versions |
| `smoke.json` | cases, recorded results, candidate and confirmed assertions, who confirmed |
| `smoke_runs.jsonl` | every smoke_verify run, per assertion |
| `probe.json` | tasks, moments, use checks, harness, tier, bounds, the gate, who set them |
| `probe_trials.jsonl` | one row per trial (3.12, E9's schema extended) |
| `verdict.json` | the gate applied, every number with its n, armed or not |

The dataset row itself stays in the jobs database (`datasets` table), its
stage jobs in `jobs` with `dataset = <row>`, as onboarding's are. No response
carries a filesystem path (`skills.public`'s rule).

---

## 7. The Tools page

A new top-level screen beside Skills. Its name is [open]: routes and pages
take bonsai terms (AGENTS.md "Naming"); `DOGU` (bonsai tools, 道具) is one
option for the operator.

**LIST** (`/tools`): one card per server -- title, `server@version`, state
chip (`ingesting`, `waiting: approval`, `waiting: tests`, `verifying`,
`armed`, `not armed`, `refused`, `drifted`, `disabled`), its tools, the
tiers, last verified, the probe's headline (should-call k/n per task, with
n), today's calls and errors. A running row shows the `StageRail` over its
stages (the onboarding card's idiom, `web/src/screens/skills/onboarding.tsx`)
and, while a gpu stage waits, "waiting for an idle stack: <why>".

**ADD**: one field for the source (npm name, PyPI name, GitHub URL, image
reference, remote URL), one line for the goal. Nothing else is asked until a
rule needs it.

**DETAIL** (`/tools/<row>`), one panel per stage, in pipeline order:

1. RESOLUTION -- the table, the licence quote and where it came from; the
   licence form when none was found.
2. APPROVAL -- the download list (file, source, size), the total, the base
   image; APPROVE / DECLINE. Shown until decided.
3. TREE -- packages, integrity, licences, declared install scripts (off),
   signatures.
4. AUDIT -- findings grouped by question, each with file:line and the
   excerpt, reachable or not; the hosts (the allowlist, with an ADD HOST
   field); the credentials it can use (a SUPPLY field per name, write-only);
   refusals in red with what would change them; the model's summary with its
   citations, and its PROMPT disclosure.
5. TOOLS -- per upstream tool, side by side: the vendor's description
   (escaped, labelled "vendor text: data, never sent to the model"), and ours
   as main will receive it (the exact function JSON); keep/drop with the
   reason and its evidence link; each lint verdict (section 5); the explore
   calls and their results; EDIT on every proposed part (an edit re-lints and
   re-opens verification); the system line.
6. TESTS -- the confirm form (3.10): the smoke table with recorded results
   and assertion checkboxes; the probe tasks, editable; moments and use
   checks as pick lists (labels); the run settings; the GATE fields, empty
   until filled, with what n can show. CONFIRM.
7. VERIFY -- smoke results per assertion; per trial: outcome, the calls
   before the moment with their arguments and returned names, the moment's
   action, each use check with its evidence, `invalid`/`not_run` with why,
   links to the relay rows; the gate verdict with every number and its n.
8. OPERATE -- enable, tiers, watch state and the next version's row, drift,
   the usage panel, the recent `x_yamadori.mcp` calls.

Every model stage's panel has a PROMPT disclosure: template id and version,
the rendered system and user text, the raw reply, the parsed proposal, the
code verdicts (section 4).

---

## 8. The API (under `/dash/api`, behind `accounts.identify`)

| call | body / answer |
|---|---|
| `GET /dash/api/mcp` | EXISTS: the configuration in force and each server's state |
| `GET /dash/api/tool-factory` | every row: {id, server, version, stage, state, updated, headline} |
| `GET /dash/api/tool-factory/<id>` | the dataset detail (jobs, blockers, warnings, next stage) plus every record of 6.2, public fields only |
| `POST /dash/api/tool-factory` | `{source, goal}` -> `{id}` |
| `POST /dash/api/tool-factory/approve` | `{id, list_sha256, decision}` |
| `POST /dash/api/tool-factory/credential` | `{id, name, value}`: write-only; the answer never echoes the value |
| `POST /dash/api/tool-factory/hosts` | `{id, add: [host]}` |
| `POST /dash/api/tool-factory/tool` | `{id, tool, fields}`: an edit; re-lints, re-opens verification |
| `POST /dash/api/tool-factory/tests` | `{id, smoke, probe, settings, gate}`: the confirm |
| `POST /dash/api/tool-factory/rerun` | `{id, stage}`: an errored stage, or a new draft version |
| `POST /dash/api/tool-factory/serve` | `{id, enabled, tiers}` -> `mcp_config.save` |
| `GET /dash/api/tool-factory/prompts` | the templates, versions and hashes |

Every write records its author (as the skill factory does).

---

## 9. What exists and what must be built

**EXISTS, reused as it is:**

| piece | where |
|---|---|
| the MCP client, one long-lived sandboxed server per row, crash restart, image check, failure envelopes, render + screen + data frame, the hidden hop, `x_yamadori.mcp`, the offer kept per conversation, overlaps | `mcp/mcp_host.py`, `mcp/proxy.py` (`_mcp_offer`, `_run_our_tool`, `tool_conflicts`) |
| the served configuration, its validation and its writers | `mcp/mcp_config.py` (`save`, `set_enabled`) |
| the gated network and the gate | `bench/sandbox/sandbox_net.py`, `egress_gate.py` |
| the harness box (Pi, OpenCode, Codex; default loadout) and a model-free MCP client inside it | `bench/sandbox/harness_box.py`, `mcp_probe.mjs` |
| the probe's measurement shape: preflight, relay, clean header, stop at the moment, invalid-run checks | `bench/mcp/lookup_probe.py`, `bench/octopus/run.py` (`pi_preflight`, relay), `bench/octopus/pagoda.py` (`pi_models`, `advertised_card`) |
| the durable queue, idle gating without spent attempts, the worker's handler registry and advance hook | `mcp/jobs.py`, `mcp/idle.py`, `mcp/worker.py` |
| datasets: stages, holds, blockers, answer, rerun, warnings | `mcp/datasets.py`, `mcp/onboarding.py` (the kind-`package` pattern) |
| resolve and licence from verbatim quotes; GET-only fetching with rate-limit deferral; tarball integrity | `mcp/package_resolve.py`, `mcp/package_net.py`, `deps.fetch_verified`, `worker.find_verbatim` |
| the screens: fetched text, credentials, shell parsing | `mcp/skill_screen.py` |
| versioned, hash-pinned templates; `DATA_PARAGRAPH`; the doubt rule; the leak rule | `mcp/skill_prompts.py`, `mcp/skill_limits.py`, `mcp/skill_prove.py` |
| the one door for the model stages | `mcp/model.py` |
| the dashboard's stage rail and onboarding card | `web/src/screens/skills/` |
| the offline fake MCP server and the host's tests | `mcp/fixtures/fake_mcp_server.py`, `mcp/test_mcp_host.py`; offline stores point `YAMADORI_MCP_SERVERS` at a temp file (`mcp/offline_stores.py`) |

**BUILD:**

| id | piece |
|---|---|
| B1 | kind `mcp_server` in `datasets` (stages, enqueue table, holds, blockers), `mcp/tool_factory.py` (handlers, records, sweep), registered in the worker |
| B2 | `servers.json` version 2 and `mcp_config.validate` for it; the PackageLens backfill |
| B3 | the MCP client: newer protocol revisions, `structuredContent` / non-text content, the generic render; resource limits in `run_argv`; the tools/list hash and drift |
| B4 | the gate's per-network host allowlist and its `DENY` findings |
| B5 | resolve for images and PyPI wheels; the tree stage (Node lockfile in a gated container; Python hash-locked wheels); the recipe writer |
| B6 | the audit: tree-sitter queries per finding kind (JS/TS, Python), the import graph from the entry, reachability, refusals |
| B7 | `mcp/tool_prompts.py` (section 4) with its pin test; the lint (section 5); the assembler |
| B8 | smoke: cases, assertion kinds, record and verify runs |
| B9 | `mcp/tool_probe.py`: lookup_probe's machinery as product code, the candidate header for test accounts, moments and use checks as label tables (with tree-sitter bash for commands), the test-labelled probe account |
| B10 | decide + arm; the watch schedule; the `mcp_call` corpus event and the usage aggregation |
| B11 | the API (section 8) and the React page (section 7) |
| B12 | tests: offline end to end over the fake server and a fake registry (the offline guard forbids network and live state); a LIVE check in `mcp/test_live_stack.py` that runs the pipeline on PackageLens and compares with the hand result (M3); `docs/LIVE-COVERAGE.md` rows |

---

## 10. Milestones, in order

The order is fix -> test -> deploy -> run (memory `fix-before-run`), and the
hand-built PackageLens result is the first acceptance test: the factory is
right when it reproduces what we did by hand.

| # | milestone | done when |
|---|---|---|
| M0 | **Fix what the evidence found** (offline): the probe's command parser (tree-sitter bash), a test-labelled probe account, `run_argv` resource limits, the `mcp_call` corpus event; the operator's answers on D6/N3 for the probed tools; PackageLens backfilled as the first factory row (E1-E11 as records, `lookup_probe.jsonl` as its probe) | offline suites green; `servers.json` carries PackageLens |
| M1 | **Records, lint, generic render** (B2, B3 part, B7's lint): offline over the fake server | a fake server's tools lint and render; PackageLens's hand descriptions produce the section 5 verdicts |
| M2 | **The model-free pipeline for npm stdio servers** (B1, B4, B5 Node, B6, B8): resolve -> approve -> tree -> audit -> build -> list -> smoke, with a hand-written draft | offline end to end; then LIVE on PackageLens: the audit's hosts equal E3's eight, no `DENY`, smoke passes E8's expectations written as assertions |
| M3 | **The model stages** (B7 templates): audit_summary, explore, draft, smoke_cases | LIVE on PackageLens, idle stack: the draft keeps the three tools E5 kept (or states evidence for a difference), finds `ecosystem` by a with/without pair, and its lint passes. A difference from the hand result is a finding to diff, not to tune away |
| M4 | **The probe and the gate** (B9, B10 decide/arm) with Pi | LIVE: PackageLens's probe re-run through `tool_probe.py` with E9's settings; the use checks read real values (no `2>&1`); `project_installs` now exercised on the post-E11 tool -- the probe E11 never had |
| M5 | **The Tools page and API** (B11) | a person adds a server, approves, confirms tests, sees the verdict, without a terminal |
| M6 | **Operate** (B10 watch, drift, usage; tiers) | a new upstream version opens a row and serves only after it arms |
| M7 | **More sources**: PyPI wheels, image-only servers, GitHub source builds, remote MCP over HTTP (credentials, no audit possible: section 12) | each with one live run on a server the operator names |
| M8 | **More harnesses for the probe**: OpenCode, Codex, Hermes (each harness's event format for moments) | the same task probed on each, per memory `harness-tools-direction` ("test and verify every harness") |

A second, genuinely new server enters at M3-M4, chosen by the operator; the
plan does not pick it.

---

## 11. Risks

| risk | evidence | what the plan does |
|---|---|---|
| the local model's drafts are wrong or doubtful | SkillsBench -8.1 to -11.5 pp for self-generated; ternary PTQ loses most on code and reasoning (SKILLS-RESEARCH.md 1.2); section 0's wrong names | parts not prose; code assembly and lint; facts never from the model; the probe is the proof |
| the probe's n is small and one prompt | E9: n=4, one prompt; PROTOCOL rules 4 and 10 | n and gate are the user's; the page states what n can show; several tasks per tool |
| a 100% call rate hides a harness defect | PROTOCOL rule 3; E9's use metric was broken in 4/4 | every trial row carries the calls, arguments and returned names; use checks are code with tests |
| over-calling (the tool fires when not for) | SRA 2604.24594; SMART [A]; not yet evidenced on Bonsai | should-not tasks in every probe |
| the tool list grows and selection degrades | "Less is More" 2411.15399; "Linearly Readable" 2605.07990; PackageLens alone adds 3,378 characters | D9 and C1 reported per server and in total; tiers per server |
| the arguments leave for any host | the gate allows any global address today | the per-server allowlist (B4); audit hosts; `DENY` findings |
| a malicious or compromised server | untrusted code in a container on the Docker Desktop VM | scripts off, signatures, audit before run, uid 1000, no capabilities, read-only root, no credential, resource limits (B3), gated network |
| vendor text as prompt injection | E3's "CRITICAL RULES"; INJECTION.md F1 | vendor text read only inside `DATA_PARAGRAPH`, never sent to main; results screened |
| a tool that needed code (peers, README fallback) | E10-E11 | the probe's use checks find the gap; the page reports it; the fix is ours or another server's |
| the probe measures through `:1234` and competes with users | AGENTS.md "One GPU consumer at a time" | idle-gated gpu stage, preflight, 429 is not run |
| probe traffic learned from | the `pi-dogfood` key is client traffic | a test-labelled account (M0) |
| arming reverts PackageLens | `mcp_config.load` drops DEFAULT once the file exists | the backfill writes PackageLens first (M0) |
| the MCP client is PackageLens's subset | protocol `2024-11-05`, text content only | B3 before a server that needs more |
| tier choice vs the operator's xhigh focus | E9 ran at medium; memory `bonsai-bet`: "our xHigh thinking is all I care to test" | the tier is a probe setting; a clean context at xhigh needs our plan/think tools off (3.12) |
| user-supplied credentials | none used so far | names only in records; the store is the operator's decision |

---

## 12. Open decisions for the operator

1. **D6 and N3 on the probed tools**: `yama_package_readme` fails the doubt
   rule ("may differ from what you remember") and two PackageLens names are
   noun-first. Bind them (reword, rename, re-probe) or grandfather them?
   And is N3 a FAIL or a REPORT for new tools?
2. **Where a user's server lives**: recipes and image records in the repo
   (`mcp_servers/<id>/`, `models/manifest.yaml`, as PackageLens) or in the
   state directory with a factory manifest the deploy check also verifies.
3. **Reachable `exec`**: a hard refusal, or a finding the user
   acknowledges.
4. **Image-only and remote servers** (M7): with no source to audit, allowed
   with an acknowledgement, or refused.
5. **Python sdists**: refuse, or build in a gated container (which executes
   their build backend).
6. **The credential store** and who may supply a credential.
7. **Explore's call count per tool** (each is a third-party request).
8. **Container resource limits** for hosted servers (harness_box's values are
   the only precedent).
9. **A new version of a tool**: serve the old until the new arms (the
   onboarding rule), or disable on drift.
10. **The page's name** (bonsai terms; `DOGU` offered).
11. **Should a user-confirmed smoke be enough to arm with no probe** when the
    user sets an empty gate? THE TOOL RECIPE says a probe before it ships; the
    plan reads that as: no.

---

## 13. Security, in one place

- Downloads only after an approval that lists file, source and size; the
  approved list's hash is recorded and a change needs a new approval.
- Build with lifecycle scripts off, every integrity and signature checked, a
  pinned base never pulled without approval (`--pull=false`).
- The source audited before the server first runs; hard refusals for
  workspace tools, unauditable binaries and unpinnable artefacts.
- Every run of the server -- list, explore, smoke, probe, serving -- in its
  own container: uid 1000, no capabilities, no-new-privileges, read-only root,
  resource limits, a gated network with a per-server host allowlist, no
  credential unless the user supplied one for that server.
- Vendor descriptions and results are data: framed, screened, never
  instructions; ours replace theirs.
- The probe runs as test traffic with a test key, in the harness box, through
  the proxy's one door.
- No response carries a path or a secret; every write records its author.
