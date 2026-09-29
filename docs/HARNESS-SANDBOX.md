# Harness sandbox: where model-driven code runs

2026-09-26. SELF-IMPROVEMENT-LOG #48 (Octopus) and #49 (SWE-bench and the
host-run harness tests). Configuration only. No harness is patched, and no
Docker, Windows or stack setting is changed. No model was run for any of
this, and nothing was restarted.

## The hole

This machine runs Docker Desktop with the WSL2 engine and `NetworkType:
gvisor`. A container on Docker's default bridge reaches the Windows host
through `host.docker.internal` = 192.168.65.254. The connection is made on
the host's side, **to the host's loopback**. Every 127.0.0.1-bound service
answered a probe from a container:

- llama-swap :11434 (no auth)
- llama-server :10001
- Caddy's admin API :2019
- the tools API :1235
- SearXNG :8888
- the proxy :1234

SMB :445 and the host's LAN, ZeroTier and WSL addresses answered too (#48).
Hiding the name does nothing, because the address answers directly.

Two more setups were worse, because the model's shell was the host itself:

- the OpenCode and Pi tests (`C:\Users\jwals\octo\opencode-test`, `pi-test`);
- the Codex home in the session scratchpad.

There the model could reach every loopback service, the whole filesystem,
and the key files that sit beside the configs. Codex's Windows sandbox
restricts writes. It does not restrict reads or loopback.

## The shared piece: `bench/sandbox/`

`sandbox_net.py` and `egress_gate.py` moved here from `bench/octopus/`. The
files at the old paths are aliases, so `import sandbox_net` in Octopus code
returns the moved module object itself. Octopus's tests are unchanged except
for the gate's mount path (`/sandbox`).

Each run unit gets its own:

- **Network.** Created with `docker network create --internal
  <prefix>-net-<tag>`. It has no gateway and no external DNS.
- **Gate.** The `<prefix>-gate-<tag>` container runs on the pinned
  `nikolaik/python-nodejs` image that is already local. It sits on both
  networks and answers as `egress` on the internal one. It is an HTTP/CONNECT
  proxy on :3128. It resolves each name itself and connects only to an
  `ipaddress.is_global` address on port 80 or 443. It logs every
  ALLOW / DENY / FORWARD.
- **Model containers.** They join only the internal network and get
  `HTTP(S)_PROXY` (both cases) = `http://egress:3128`. Anything that ignores
  the proxy variables has no route.
- **Cleanup.** `down()` force-removes any container still on the network,
  then the gate, then the network. It records the allowed, denied and
  forwarded counts.

Prefixes: `octo` (Octopus), `swe` (SWE-bench), `harn` (harness box).

`python bench/sandbox/sandbox_net.py verify [--image I] [--tools npm,pip,git]`
is the live check. It uses Docker and no model. It runs the probe in a
container that gets exactly the args a model container gets:

- direct TCP to every host address × {1234, 1235, 8888, 11434, 10001, 2019,
  445, 80, 443};
- plain-HTTP and CONNECT requests to each host target through the gate;
- the npm registry, pip from PyPI, and `git clone` over https.

## SWE-bench (mini-swe-agent)

- `bench/swebench/wsl_side.py` `cmd_agent` brings the network up for each
  instance (`swe-net-<instance>-<pid>`) before mini starts. It brings it down
  after mini exits, including when mini raises. If there is no gate, mini does
  not run: `rc` is 97 and `not_run: sandbox_net` is recorded in
  `timings.jsonl` and in `parse_results`' row.
- The overlay adds two keys: `environment.run_args` (`--rm --network <net>`)
  and `environment.env` (the proxy variables, merged into the yaml's env by
  mini's `recursive_merge`). Both are listed in `OVERLAY_DEVIATIONS` and
  `docs/SWE-BENCH.md`. The stock `docker` class applies `run_args` as they
  are, which covers the WSL path.
- `dockerfix.LowLevelDockerEnvironment` creates its container with
  `network_mode` = that `--network`. When there is none, it uses `none`.
  `bridge` and `host` are refused.
- This deviates from the leaderboard, which ran with an open network. The
  model can see the proxy variables.

## The harness box: OpenCode, Pi, Codex in a container

`bench/sandbox/harness_box.py` + `bench/sandbox/harness/` (Dockerfile,
package.json, package-lock.json).

### The network

It is the shared sandbox, with one addition: **exactly one forward to the
host**. The forward is `egress:<port>` → `host.docker.internal:<port>`, where
`<port>` is the proxy (:1234) or the recording relay the test goes through
(the host tests used 127.0.0.1:18235 and :18236).

- The runner fixes the target. Nothing a client sends can choose it.
- The forward is not published on the host.
- `--target-port` refuses the service ports #48 found open (11434, 10001-10008,
  2019, 1235, 1237, 8888, 445, 139, 135, 2222, 80, 443).
- `egress` is in `NO_PROXY`, so the harness's API calls go straight to the
  forward.
- Everything else goes through the gate: public addresses on 80/443 only.

### The container

With the default loadout (below), the harness container joins the browser
sidecar's network namespace, `--network container:harn-browser-<tag>`, and
the sidecar is the one on `harn-net-<tag>` as `harness`. `--loadout
as-tested` gives the original form:

```
docker run --rm -i --init --network harn-net-<tag> --network-alias harness
  --user 1000:1000 --cap-drop ALL --security-opt no-new-privileges
  --pids-limit 1024 --memory 8g --cpus 4
  -v <project>:/work -v <run-dir>\home:/home/node -w /work
  -e HTTP(S)_PROXY / NO_PROXY=...,egress  -e NODE_USE_ENV_PROXY=1  -e HOME=/home/node
  -e <the host test's own variables>  -e YAMADORI_<H>_KEY
  yamadori-harness-box:oc1.18.32-pi0.87.1-cx0.157.1-tools1 <opencode|pi|codex> <args>
```

The environment variables are the same ones the host tests set:

- **OpenCode:** the four XDG dirs, `OPENCODE_CONFIG`,
  `OPENCODE_DISABLE_AUTOUPDATE`, `_SHARE`, `_CLAUDE_CODE`
  (`docs/HARNESS-OPENCODE.md`).
- **Pi:** `PI_CODING_AGENT_DIR`, `PI_CODING_AGENT_SESSION_DIR`,
  `PI_OFFLINE=1`, `PI_SKIP_VERSION_CHECK=1`, `PI_TELEMETRY=0`
  (`docs/HARNESS-PI.md`).
- **Codex:** `CODEX_HOME` (`docs/HARNESS-CODEX.md`).

Only two things are mounted: the project, and a per-run home that holds the
harness's config, sessions and caches. Nothing else from the host is
visible.

### The config

The runner writes the config into the run home from the host test's own
file. It never mounts the host config:

| harness | source | written | base URL | key |
|---|---|---|---|---|
| OpenCode | `octo\opencode-test\cfg\full.json` | `.config/opencode/opencode.json` | `http://egress:<port>/v1` | `{env:YAMADORI_OPENCODE_KEY}` |
| Pi | `octo\pi-test\cfg\full.json` | `.pi/agent/models.json` | same | `$YAMADORI_PI_KEY` (never a `!command`) |
| Codex | `<scratchpad>\codex-home\` (`config.toml` + `yamadori-catalog.json` only; `auth.json` is never copied) | `.codex/` | same | `env_key = "YAMADORI_CODEX_KEY"` |

- Any literal key or auth header in the source is dropped. `setCacheKey`
  and the other options are kept.
- Autoupdate and share are off.
- Codex's `model_catalog_json` is rewritten to its container path.
- Before the container starts, the runner scans the run home and the project
  for the key. If it finds it, the run is not started.

### The key

The key is read from `--key-file` into the environment of the `docker run`
child only. It is passed as `-e NAME`, so the value is never in an argv, a
file the runner writes, or `harness_box.jsonl`. It is not in the image.

**OpenCode and Pi: the model can read it.** Their bash tool inherits the
harness's environment. A scripted run showed `env | grep -c
YAMADORI_<H>_KEY` = 1 in both (see "Verified" below). Neither has a config
key that filters its shell's environment, so use a key made for these tests.
Keeping the key out of the container altogether would mean the gate injecting
the `Authorization` header into forwarded requests. That was not built.

**Codex: it does not.** On Linux, Codex writes a shell snapshot of its
environment, key included, to `CODEX_HOME/shell_snapshots/`. That is on the
host's disk, through the run-home mount. Codex then sources the snapshot into
every command. The runner writes two keys into Codex's config:

- `features.shell_snapshot = false`
- `shell_environment_policy.exclude = ["YAMADORI_CODEX_KEY", "YAMADORI_*"]`

With the scripted stand-in (below) the model's `env | grep -c` was 1 with
neither key, 1 with either one alone, and 0 with both. With both, no file in
the run held the key. Codex itself still reads the key from its own
environment. This deviates from the host test's config. On the host
(Windows) no shell snapshot was ever written.

### The image

`harness/Dockerfile`:

- **Base.** `node:24.21.0-bookworm@sha256:64af3819...` (Node 24 LTS), the
  index digest read from Docker Hub with `docker buildx imagetools inspect`,
  which fetches metadata only. It is the full image, not `-slim`, because the
  full image already has git, curl, ca-certificates and python3, so nothing
  comes from an unpinned apt mirror.
- **Install.** `npm ci --omit=dev --ignore-scripts`, then `npm rebuild
  opencode-ai`. That runs only opencode-ai's own postinstall, which puts the
  linux binary at `bin/opencode.exe`, its `bin`.
- **ripgrep.** Codex's vendored `rg` 15.2.0 is linked onto `PATH`. It is
  pinned through the lockfile's sha512. The host tests had `rg` on `PATH`
  (scoop) and no `fd`, and so does the box.
  - Pi runs with `PI_OFFLINE=1`, as on the host, so it never downloads rg or
    fd. Without this link, its grep tool fails.
  - OpenCode uses the `rg` on `PATH` instead of fetching its own.
- **User.** It runs as `node`.

`package.json` pins the exact versions: `opencode-ai` 1.18.32,
`@earendil-works/pi-coding-agent` 0.87.1, `@openai/codex` 0.157.1.

`package-lock.json` was generated inside a sandboxed container
(`npm install --package-lock-only --ignore-scripts`, npm 10.9.9):

- 164 packages, all from registry.npmjs.org, each with a sha512.
- Five Pi workspace packages (`@earendil-works/chord`, `pi-agent-core`,
  `pi-ai`, `pi-telemetry`, `pi-tui` 0.87.1) came through Pi's
  npm-shrinkwrap without an integrity. Their `dist.integrity` was filled
  from the registry, after checking that the tarball URL matched.
- The linux-x64 binaries of OpenCode and Codex are pinned the same way.

`harness_box.py build` checks for the base image in the local store first.
If it is missing, the build stops with exit code 2 and prints what it would
pull. It never pulls. It builds with `--pull=false`, and the Dockerfile runs
all three `--version` commands (and `rg --version`) at the end of the build.

**Built 2026-09-26.** The operator pulled the base at that digest (local id
`sha256:b795e77f...`).

- The image was `yamadori-harness-box:oc1.18.32-pi0.87.1-cx0.157.1`, id
  `sha256:94a13239f2d30b253ff44a05d4a831d1cb56099355e3ebca8cfeffa5475f9e22`,
  2.7 GB.
- **Rebuilt on 2026-09-27 at 02:53Z with the default loadout.** The new
  image is `yamadori-harness-box:oc1.18.32-pi0.87.1-cx0.157.1-tools1`, id
  `sha256:e5ce5d268b61f6d7eb215beec2d8779249e87d77ff49c207b95a45f6eab396b7`,
  2.8 GB.
  - `npm ci` installed 158 packages of the lockfile's 266.
  - The recipe hashes and the id are in models/manifest.yaml.
- `npm ci` installed 151 of the lockfile's 164 packages. The rest are
  OpenCode's and Codex's builds for other platforms.
- Pi's shrinkwrap installs all 26 `@esbuild/*` platform binaries, 285 MB.
- The recipe (Dockerfile, package.json, package-lock.json, by sha256), the
  image id and the base are recorded in `models/manifest.yaml` `runtimes:
  harness-box`. `scripts/verify_artifacts.py` checks the recipe files.
  `harness_box.py verify` checks the local image id against the recorded one,
  and every run record carries `image_id` and `base_image`.
- The build is source-reproducible, not bit-for-bit: `npm ci` writes fresh
  timestamps. A rebuild without the layer cache gets a new id, which must
  then be recorded.
- `bench/sandbox/harness/.gitattributes` (`-text`) keeps a CRLF checkout from
  changing the pinned bytes.

### The default loadout (2026-09-26)

Operator, 2026-09-26: every harness gets usable browser tools, a normal
coding loadout and language-server diagnostics, by default. The full
per-harness list, the tool-list sizes and the bridge vetting are in
docs/HARNESSES.md §0. `harness_box.py run` writes the loadout into the run
home, on top of the host test's config. `--loadout as-tested` writes the
host test's config alone, with no sidecar.

**The browser runs in a SIDECAR.** It is the grader's own
`octo-playwright:1.63.0` image (HeadlessChrome 153.0.8010.12; no new image,
no download), running `bench/octopus/browser_sidecar.py`, the same script
as the Octopus sidecar.

- **Placement.** The sidecar is started after the gate, on `harn-net-<tag>`
  as `harness`. The harness container joins its namespace, so `localhost`
  is one place for the model's shell, its dev server and the browser.
- **Hardening.** The sidecar runs with `--cap-drop ALL`,
  `no-new-privileges`, a pids limit, 4 GB and 2 CPUs. It publishes
  nothing.
- **What the page reaches.** Chrome's proxy is the gate
  (`SIDECAR_PROXY=http://egress:3128`; the Octopus sidecar keeps its dead
  proxy). The page therefore reaches the box's loopback directly, and
  public 80/443 through the gate. It never reaches the host.
- **DevTools.** Chrome's DevTools port is 127.0.0.1:9322, inside the
  namespace. The harnesses attach there:
  - OpenCode and Codex through the official Playwright MCP server
    (`@playwright/mcp` 0.0.82, Microsoft, with a SLSA provenance
    attestation). It is launched by the harness as a stdio MCP server
    inside the box, with `--cdp-endpoint http://127.0.0.1:9322
    --no-webmcp`, so it downloads and launches no browser.
  - Pi through the `agent-browser` 0.26.0 CLI (Pi has no MCP client).
- **Lifecycle.** The sidecar is removed after the harness exits, before the
  network. The run record has `browser_sidecar` (image id, Chrome version,
  Chrome starts). A sidecar that does not answer means the harness never
  starts (`not_run: browser_sidecar`).

**The image** (rebuilt, see "The image" above) adds these, each pinned in
`package-lock.json` by sha512:

- `@playwright/mcp` 0.0.82;
- `typescript-language-server` 6.0.1, `typescript` 5.9.3 and `pyright`
  1.1.414, for OpenCode's lsp, and as `tsc` / `pyright` on `PATH` for the
  others;
- `agent-browser` 0.26.0.

It also adds fd 10.5.0, the official release, fetched with `ADD
--checksum=sha256:` against the digest GitHub records for the asset.

The build now runs `npm audit signatures`: 158 packages have verified
registry signatures and 64 have verified attestations, or the build fails.

The configs, per harness:

- **OpenCode.**
  - The six tools plus `lsp`; the rest denied.
  - `playwright_*` denied, then 13 chosen browser tools allowed.
  - `formatter: false`.
  - Every built-in language server disabled, plus `typescript-box` and
    `pyright-box` from the image.
  - Environment: `OPENCODE_DISABLE_LSP_DOWNLOAD=1` and
    `OPENCODE_EXPERIMENTAL_LSP_TOOL=1`.
- **Pi.**
  - `settings.json` `defaultTools` = `read, bash, edit, write, grep, find`.
  - The skills `page-check`, `type-check` and (2026-09-29) `package-api`
    in `~/.agents/skills`; `write_home` copies each skill's whole folder
    (`package-api` carries its script, `api.cjs`).
- **Codex.**
  - `[mcp_servers.playwright]` with `enabled_tools` (the same 13) and
    `default_tools_approval_mode = "approve"`. Without it,
    `approval_policy = "never"` refused every non-read-only MCP tool.
  - `[sandbox_workspace_write] network_access = true`: a dev server inside
    Codex's bwrap sandbox must be reachable by the browser. This opens the
    box's network (localhost, the gate, the forward) to Codex's shell, as
    OpenCode's and Pi's already have it. The key stays excluded from that
    shell.
  - The skill `type-check` in `$CODEX_HOME/skills`.

**Checked with no model:** `python bench/sandbox/loadout_check.py all
--key-file <dummy>`. A scripted stand-in (`scripted_model.py`, on
127.0.0.1, behind the one forward; both wires) makes the tool calls a model
would make, and each harness executes them in the box. Results, 2026-09-27,
on the recorded image, one row per harness in
`bench/sandbox/results/loadout_check.jsonl`:

| | OpenCode | Pi | Codex |
|---|---|---|---|
| tools / chars | 20 / 24,638 | 6 / 4,689 | 21 / 15,476 |
| shell serves localhost:5173; host refused (gate 403, direct 000) | yes | yes | yes |
| browser opens it (title), reads `console.error` and the uncaught `ReferenceError` | yes | yes | yes |
| browser to `host.docker.internal:11434`, `192.168.65.254:1234` | 403 from the gate | "egress gate: ... refused" | 403 from the gate |
| browser to https://example.com through the gate | Example Domain | Example Domain | Example Domain |
| type errors | `write` result: `LSP errors detected ... TS2322`; pyright `reportArgumentType` | `tsc`: TS2322; `pyright`: reportArgumentType | the same, through `exec_command` |
| key in the model's shell | 1 (known) | 1 (known) | 0 |
| checks passed | 10/10 | 11/11 | 9/9 |
| 2026-09-29, with the package-API steps (koota 0.6.6 copied into the project) | 12/12: `lsp` hover `(alias) function createWorld(options: WorldOptions): World (+1 overload)`, goToDefinition `node_modules/koota/dist/index.d.ts` | 19/19: `package-api` printed both `createWorld` overloads and `World.query`, the entry points, a miss; plus screenshot + `read` as image, the skills listed, `prompt_cache_key` | not rerun |

The gate's record for each run shows 3 denials, exactly the three host
probes. It allowed only `example.com`, plus OpenCode's own
`registry.npmjs.org` and `models.opencode.ai` at startup.

`harness_box.py verify --target-port 18299`, run against a throwaway
listener standing in for the relay, passed 16/16. That includes
`loadout_versions`, which checks `playwright-mcp`, `typescript-language-server`, `tsc`,
`pyright`, `agent-browser` and `fd` at their pinned versions, and
`image_is_the_recorded_one`.

**Found on the way:**

1. agent-browser 0.26.0 prints `✗` with no text for each page error
   (`errors`); `errors --json` has the text. The skill says `--json`.
2. Codex: without `default_tools_approval_mode = "approve"`, MCP tool calls
   were refused under `approval_policy = "never"`.
3. The first language-server bridge tried, `cclsp` 0.7.0, reported files
   with type errors as clean, so it was taken out of the image
   (HARNESSES.md §0).

### Commands

```
python bench/sandbox/harness_box.py build
python bench/sandbox/harness_box.py plan opencode --target-port 18235 [--loadout as-tested]
python bench/sandbox/harness_box.py verify [--target-port 1234]      # Docker only, no model
python bench/sandbox/harness_box.py run opencode --key-file K --project DIR --run-dir DIR \
    --target-port 18235 -- run --format json -- "prompt"
python bench/sandbox/harness_box.py run pi    ... -- --mode json -p "prompt"
python bench/sandbox/harness_box.py run codex ... [--codex-sandbox off] -- exec --json -C /work "prompt"
python bench/sandbox/loadout_check.py all --key-file <dummy> [--port 18299]   # the loadout, no model
```

`run` uses the default loadout unless `--loadout as-tested` is given.
`verify --target-port 1234` sends one `GET /health` to the proxy through
the forward. Point it at a stand-in port instead (`--target-port 18299`
with any listener) when nothing may reach `:1234`.

Options:

- `--timeout S` removes the container when it expires.
- `--stdin` passes stdin through. There is no stdin by default: with `-i`
  and an open pipe, `opencode run` and `pi -p` wait for EOF and never start.
  That was seen here; both hung.
- `--codex-sandbox`: `keep` (the default) or `off`. See below.

From Git Bash, set `MSYS_NO_PATHCONV=1`. Otherwise `-C /work` reaches Codex
as `C:/Program Files/Git/work`, and Codex fails with "No such file or
directory".

Every run appends a record to `<run-dir>/harness_box.jsonl`:

- the argv (names only);
- `image_id` and `base_image`;
- `codex_sandbox`, for Codex;
- the network record;
- the gate's allowed/denied/forwarded counts and `allowed_hosts`;
- the rc, and `timed_out` if the timeout fired.

### Codex's sandbox inside the box: `keep`

Codex's Linux sandbox is its vendored `bwrap`. It starts inside the
container under `--cap-drop ALL` and no-new-privileges.

The check used a scripted stand-in for the model, with no model behind it.
Codex made its own `exec_command` call through the agent path, and the
command ran in the box:

| | `keep` (workspace-write) | `off` (danger-full-access) |
|---|---|---|
| write `/work` | yes | yes |
| write `/home/node` | **blocked** | yes |
| reach `egress:<port>` (the forward) | **blocked** (Codex's sandbox has no network) | 200 |
| uid | 1000 | 1000 |

`keep` is the default and what the host test ran. The host's Windows sandbox
also had the network off. `off` stays available, and is recorded in the run
record when used.

## Verified (2026-09-26, Docker, no model)

- **Octopus.** `python bench/octopus/sandbox_net.py verify` (through the
  alias, the moved gate):
  - 0 of 54 host address:port pairs open.
  - 20/20 host targets refused by the gate for plain HTTP, and 20/20 for
    CONNECT.
  - The npm registry and PyPI answered 200.
  - `npm install is-number@7.0.0` returned rc 0.
  - `pip install six==1.16.0` returned rc 0, and six 1.16.0 imported.
  - `git clone --depth 1 https://github.com/octocat/Hello-World.git`
    returned rc 0, at commit 7fd1a60b.
- **SWE-bench.** `<swebench venv>\Scripts\python.exe
  bench/swebench/dockerfix.py netcheck` ran with the sympy image. The probe
  went through `LowLevelDockerEnvironment.execute`, with the overlay's
  environment merged onto the leaderboard yaml's.
  - The container joined `swe-net-netcheck-<pid>`.
  - 0/54 host pairs open, and 20/20 + 20/20 refused by the gate.
  - PyPI answered 200, pip install worked, and git over https worked.
  - PASSED.
  - `dockerfix.py selftest` (network `none`) passed 12/12 with
    `PYTHONUTF8=1`. Without that variable, the `unicode` case crashes the
    Windows console encoding. That was already so before this change.
- **Harness box, network** (`harness_box.py verify`, first with the gate
  image standing in, then with the built image; the same results both times):
  - curl to `http://egress:1234/health` returned 200.
  - curl direct to `host.docker.internal:11434` and `:2019` returned 000
    (the name does not resolve). Direct to `192.168.65.254:11434` and `:2019`
    also returned 000 (no route).
  - The same four targets through the gate returned 403.
  - `egress:11434`, `:2019`, `:1235`, `:80` and `:443` refused the
    connection.
  - Probe: 0/54 host pairs open, and 20/20 + 20/20 refused by the gate,
    including `host.docker.internal:1234` and `192.168.65.254:1234`. The
    target is reachable only through the forward.
  - npm and git over https worked.
  - Run again with `--target-port 18299` against a throwaway listener bound
    to **127.0.0.1 only** (the recording relay's case): the forward answered
    200. The port was closed directly from every host address (0/60 pairs
    open), and the gate refused it by name and by address (22/22 + 22/22).
- **Harness box, image.** With the built image:
  - `--version` returned 1.18.32, 0.87.1 and `codex-cli 0.157.1`.
  - `image_is_the_recorded_one` was true.
  - All 15 verdicts were true.
- **`--ignore-scripts` breaks nothing.** The image has four install scripts:
  - `opencode-ai`'s runs, through `npm rebuild`.
  - `@google/genai`'s `preinstall` is `echo 'preinstall: no-op'`. Its
    `prepare` never runs for a registry install.
  - `protobufjs`'s postinstall only warns about the version scheme its
    dependents use.
  - `esbuild`'s `install.js` only optimises the lookup of its platform
    binary.
  - Exercised in the image: esbuild 0.28.2 `transformSync` and `build`
    (Pi's chord uses the build API) worked, `require("protobufjs")` worked,
    and `import("@google/genai")` worked.
- **Every harness end to end, with no model behind it.** A stand-in for the
  relay, bound to 127.0.0.1:18299, took the one forward. Two kinds were used:
  - a logging one that answers 503 to every POST;
  - a scripted one that answers the first turn with ONE tool call running a
    probe command, and then "ok".

  In each run, the harness's own requests carried the key from the env
  reference: `Authorization` equalled the dummy key in 17 of 17 logged
  requests.

  | | OpenCode | Pi | Codex |
  |---|---|---|---|
  | its API calls | 9 POST `/v1/chat/completions` (with retries, and a title call) | 4 POST (1 + 3 retries) | 2 POST `/v1/responses` |
  | outward traffic at startup | `registry.npmjs.org:443` (it installs `@opencode-ai/plugin@1.18.32` into its config dir in the background; that finished in the 70 s run, and had not started in a 3 s one), `models.opencode.ai:443` (its model catalog) | none (`PI_OFFLINE=1`) | none (plugins/apps off) |
  | denied by the gate at startup | 0 | 0 | 0 |
  | the model's shell (scripted tool call) | uid 1000; writes `/work`; `npm view is-number@7.0.0` = 7.0.0 through the gate; `host.docker.internal:11434` and `192.168.65.254:11434` refused by the gate (403); rg 15.2.0; **the key IS in its env** | the same | uid 1000; `/work` writable, home blocked, no network (bwrap); **the key is NOT in its env** |
  | the key on disk afterwards (run home and project) | none | none (`auth.json` is `{}`) | none, with `shell_snapshot = false` (with snapshots on, `shell_snapshots/*.sh` held it) |

  All three runtime paths honour the proxy variables. Every outward
  connection went through the gate, and 0 were refused. Anything that
  ignored the proxy would have had no route, and none failed that way.
- **Offline tests.** `bench/sandbox/test_harness_box.py` 39/39 (configs,
  Codex's snapshot/env-policy keys, key handling, argv without stdin, the
  single forward, fail-closed, the build's refusal, the image pins and the
  manifest entry, `down()`'s cleanup). `bench/swebench/test_sandbox.py`
  13/13. `bench/octopus/test_sandbox_net.py` 39/39 and `test_toolset_arms.py`
  47/47. `run_tests.py` now collects `bench/sandbox/test_*.py`.
- **Real host configs.** The runner wrote configs from all three into a
  scratch home. None of the 6 key files found (scratchpad and `octo\`)
  appears in them.

## Not verified / owed

1. **A real model run in the box.** None was run. Every check above used a
   stand-in, and nothing reached `:1234` except `GET /health`. The next step
   is the harness tests themselves through `harness_box.py run
   --target-port <relay>`.
2. **One Codex start failed** out of 9 box starts on a fresh run home: "thread-store
   internal error: File exists (os error 17)". The other 8 were fine
   (n=1). It came right after rm -rf of the same run dir. A guess, not checked: the Windows bind mount under Codex's
   SQLite stores.
3. **The image is 2.7 GB.** Pi's shrinkwrap pulls in every esbuild platform
   binary. Not trimmed.

## What still runs on the host

- **The harness tests as documented.** `docs/HARNESS-OPENCODE.md`,
  `-PI.md` and `-CODEX.md` still describe host runs. They are closed only
  when rerun through `harness_box.py run` with `--target-port` = the relay.
  Until then, a model shell in those setups is the host.
- **SWE-bench grading.** The swebench harness's eval containers run the
  model's patch and the tests on the **default bridge**. Moving them changes
  the grading environment of past runs, so it needs a paired re-grade. Open
  in #49.
- **Harness processes.** mini-swe-agent and Hermes run on the host. Their
  model-directed commands run in the sandboxed containers. Hermes' own
  non-terminal tools (files, vision) run in the Hermes process: not examined
  here.
- **Octopus `--browser-host host`.** Opt-in. The host browser can navigate to
  loopback (#47).
- **Hermes' agent-browser.** The Octopus default loadout drives the
  sidecar's Chrome from the host, through agent-browser 0.26.0 in
  `hermes-home\node`, over CDP on 127.0.0.1:9222. The page itself runs in
  the sandbox. `toolset_arms.py verify` checked the whole path in Docker
  with Hermes' own tool code: 8/8, no model.
- **The relays.** `relay.py` and `capture_relay.py` are our code and run on
  the host. The forward reaches exactly that one port.
- **Pre-existing containers on the bridge.** `hermes-9399c972` (exited),
  `octo-keepalive` (up 47 h), and `nostalgic_maxwell` (created). Left for the
  operator.
