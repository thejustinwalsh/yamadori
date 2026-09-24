# Web search: SearXNG

The "web search" row of `docs/SELF-IMPROVEMENT-PLAN.md` Phase 0.6: a
self-hosted SearXNG on loopback, JSON output, developer-leaning engines,
supervised by the watchdog.

Status on 2026-09-24 17:10:
- **Running natively on Windows**, in its own venv, on `127.0.0.1:8888`, started
  by the watchdog's own entry (`scripts/watchdog.ps1`, `searxng`).
- **Called by deep thinking's `search_web` tool** (`mcp/research_tools.py`,
  Phase 0.6; built offline 2026-09-24, not yet run live). It reads
  `YAMADORI_SEARCH_URL` (loopback only), asks `format=json` and, for a code
  question, `categories=general,it`; at most 3 searches per deep-thinking run
  (`YAMADORI_SEARCHES_PER_RUN`, from the rate limits below); a query carrying
  code or something secret-shaped is refused before it leaves the machine;
  `unresponsive_engines` is reported as a partial result, never an error; a
  service that is down returns `SEARCH_UNAVAILABLE`, retryable, with the
  remedy.
- **Fetched content is data** (operator, 2026-09-24). Every title and
  snippet goes through `skill_screen.screen_fetched` (the one screen) before
  the second brain sees it: an offending span is stripped and recorded, a
  result stripped past 25% is dropped. Only a URL a search returned in the
  same run, or one the user gave, may then be read (`read_web_page`), each
  hop pinned to a global address. What deep thinking hands back is screened
  again (`screen_handoff`): a command, install step or URL from the web is
  never passed on as an instruction, only as a fact labelled "(from the
  web, unverified)".
- **The proxy has not been restarted**, so its environment does not have
  `YAMADORI_SEARCH_URL` until its next restart (by the watchdog or
  `start-stack.bat`).

```
YAMADORI_SEARCH_URL=http://127.0.0.1:8888
```

---

## How it runs

| what | where |
|---|---|
| source | `C:\Users\jwals\searxng\src`, the GitHub `master` tarball at commit `3cd69d30e` (2026-09-23), recorded in `src\searx\version_frozen.py` |
| venv | `C:\Users\jwals\searxng\.venv`, Python 3.13.15 (see "Python version") |
| config you edit | `C:\Users\jwals\searxng\settings.template.yml` |
| config it reads | `C:\Users\jwals\searxng\etc\settings.yml`, rendered from the template; holds the secret key |
| limiter config | `C:\Users\jwals\searxng\etc\limiter.toml` |
| Windows shims | `C:\Users\jwals\searxng\windows_shims\pwd.py`, on the path via `.venv\Lib\site-packages\yamadori_windows_shims.pth` |
| command | `C:\Users\jwals\searxng\.venv\Scripts\python.exe -m searx.webapp`, with `SEARXNG_SETTINGS_PATH=C:\Users\jwals\searxng\etc\settings.yml` |
| logs | `logs\searxng.log` and `logs\searxng.log.err` in this repo |
| health | `GET http://127.0.0.1:8888/healthz` answers `OK` |

The server is the Flask/werkzeug threaded server that `python -m searx.webapp`
runs. Upstream calls it a development server; upstream's production server is
granian. Granian **does** start on Windows (tried on :8890), but it runs the
app in a `multiprocessing` child whose command line is
`python.exe -c "from multiprocessing.spawn import spawn_main; ..."`, which
names neither granian nor searx. The watchdog finds and kills a service by
its command line (`Match`), so it could not stop that child, and a stale child
holding :8888 is exactly the "replacement fails to bind" case the watchdog
comment warns about. The werkzeug server is one process tree (the venv
launcher and its child), and both match `searx\.webapp`. For one loopback
client, the dev server is enough. Switch to granian only with a watchdog
change that kills the tree.

### Python version

The operator asked for 3.11 or 3.12. None is installed: the py launcher
knows only 3.9 (too old; SearXNG requires >= 3.10), and uv lists a 3.11.16
it can no longer find. Getting 3.11 or 3.12 would mean downloading an
interpreter, which was not among the approved downloads. So the venv was
made from the stack's conda base, 3.13.15, the way `.venv-laya` is made. It
is a separate venv (`include-system-site-packages = false`) and shares no
packages with the stack. Every requirement installed from a wheel. To move to
3.12 later, run `uv python install 3.12`, then
`uv venv --python 3.12 C:\Users\jwals\searxng\.venv`, then reinstall (see
"Updating").

### What failed on Windows, and the workaround

Windows is not an upstream target. These are the failures seen on
2026-09-24, in order:

1. **`git clone` cannot check out.** Four files under `utils/templates/` have
   `:` in their names (`searxng.ini:socket`, `searxng.conf:socket`), and
   Windows git refuses the checkout. Sparse checkout does not help, because
   git validates the index path before sparse filtering applies. Workaround:
   download the tarball and extract it with `tar --exclude='*:*'`. Those files
   are uwsgi/nginx/apache templates that we do not use. tar also failed to
   make one symlink (`utils/templates/etc/apache2`), which we do not use
   either.
2. **`ModuleNotFoundError: No module named 'pwd'`** at startup.
   `searx/valkeydb.py` imports the POSIX-only `pwd` at module level. It uses it
   only to name the user in the error log when a Valkey connection fails, and
   the same line also calls `os.getuid()`, which Windows lacks. We run
   without Valkey (`valkey.url: false`), so that line is never reached.
   Workaround: a stub `pwd` module in `windows_shims\`. The upstream source
   is not patched.
3. **No version string.** Without `.git`, `searx.version` logs two ERRORs
   and reports no version. Workaround: `src\searx\version_frozen.py`, the
   file upstream's packaging writes. **Update it whenever the source is
   updated.**
4. **A warning, not a failure:** curl_cffi reports `Proactor event loop does
   not implement add_reader`, and registers an extra selector thread
   instead. It works. Its effect on latency was not separated out (see the
   cold first query below).

WSL, the fallback, was not needed.

---

## Configuration

**Edit `settings.template.yml`, never `etc\settings.yml`.** Then run:

```
C:\Users\jwals\searxng\.venv\Scripts\python.exe C:\Users\jwals\searxng\render_settings.py
```

This keeps the existing key. Pass `--rotate` to generate a new one. The key
comes from `secrets.token_hex(32)` on this machine and is never printed. The
first key was rotated the same day, because a tool echoed it into an agent
transcript. That is why rendering goes to a file nothing opens. Restart
SearXNG after rendering (next section).

The settings that matter:

| setting | value | why |
|---|---|---|
| `server.bind_address` / `port` | `127.0.0.1` / `8888` | loopback only; verified: the one listener is `127.0.0.1:8888` |
| `search.formats` | `[html, json]` | JSON for the stack; html for looking at it in a browser on this box |
| `server.limiter` | `false` | single tenant; `etc\limiter.toml` also puts `127.0.0.0/8` and `::1` in `pass_ip`, so turning the limiter on cannot block our calls |
| `server.public_instance` | `false` | |
| `search.autocomplete` | `""` | autocomplete would send partial queries to a third party |
| `search.favicon_resolver` | `""` | the same, for every result's host |
| `general.enable_metrics` | `true` | in-process counters behind `/stats`, on loopback only. SearXNG sends nothing anywhere: it has no telemetry |
| `valkey.url` | `false` | no Valkey; see failure 2 above |
| `outgoing.request_timeout` | `4.0` s (max 8.0) | a choice, not measured; upstream's default is 3.0 |

---

## Engines

`use_default_settings.engines.keep_only` drops every upstream engine not
listed. It removes them outright, not just disables them.

| engine | runs when | notes |
|---|---|---|
| duckduckgo | every search | CAPTCHA'd after 3 quick queries on 2026-09-24 |
| brave | every search | 429 after ~10 queries in ~2 min |
| bing | every search | `weight: 0.5`, a choice from **one** query where it ranked three.com and "3 (company)" 2nd and 5th |
| wikipedia | every search | exact-title lookup (`Three.js` hits, `three.js` does not); results list and infobox |
| stackoverflow | every search, `it` | Stack Exchange API, title search: low recall (1 hit for "InstancedMesh setMatrixAt") |
| mdn | every search, `it` | broad matching; often ranks near the top with loosely related pages |
| superuser, askubuntu | `it`, `!su`, `!ubuntu` | |
| github | `it`, `!gh` | repository search, not code |
| npm | `it`, `!npm` | via api.npms.io |
| crates.io | `it`, `!crates` | |
| docs.rs | `it`, `!docsrs` | ours, not upstream: an `xpath` engine over `docs.rs/releases/search`, because docs.rs has no JSON search API. It breaks if docs.rs changes its markup |

**Tried and dropped:** `pypi`: pypi.org/search now answers with a
3,038-byte browser-challenge page, so the engine returns 0 results every
time. There is no key-free PyPI search API. `lib.rs`: 149 results for "serde"
in one call, which drowned every other engine in `categories=it`; crates.io
and docs.rs cover Rust. Google, Startpage and Mojeek stay off: upstream
marks Google disabled, and it marks the other two inactive because of
proof-of-work CAPTCHAs.

### Calling it

```
GET http://127.0.0.1:8888/search?q=<query>&format=json
GET http://127.0.0.1:8888/search?q=<query>&format=json&categories=general,it
GET http://127.0.0.1:8888/search?q=!crates+<query>&format=json
```

The JSON has `results[]` (`url`, `title`, `content`, `engine`, `engines`,
`score`), `infoboxes[]`, `suggestions[]` and `unresponsive_engines`, a list of
`[engine, reason]`. A 200 with some engines unresponsive is normal. Read that
field rather than treating a thin result list as "nothing exists".

---

## Measured (2026-09-24, n small; labelled as such)

`scratchpad verify_search.py`: urllib from this box, wall time per request.

| query | params | run 1 | run 2 | run 3 | results (URLs) |
|---|---|---:|---:|---:|---|
| `three.js InstancedMesh setMatrixAt` | none | 4.01 s (first query after start) | 0.75 s | 0.83 s | 44 / 45 / 44, all with URLs |
| same | `categories=general,it` | 0.63 s | 0.60 s | 0.70 s | 44 / 21 / 22 (brave 429 from run 2) |
| `rust extern C struct repr(C) wasm` | none | 0.82 s | 0.71 s | 0.76 s | 30 / 24 / 21 (brave suspended) |
| `wasm-bindgen` | `categories=it` | 0.93 s | | | 114 from 6 engines |

For the verification query, the top result in all three runs was
`https://threejs.org/docs/pages/InstancedMesh.html`, agreed by brave, bing and
duckduckgo. An earlier instance measured the same query at 3.28 / 0.58 /
0.31 s. So a warm query takes 0.3–0.9 s (n=11 warm requests on two
instances). The **first query after a start costs 3–4 s**, and on the first
instance it timed npm and lib.rs out (cold TLS + HTTP/2 connection setup);
they answered on the next call. Latency is dominated by the slowest engine
that answers, capped by `request_timeout`.

---

## Restarting

- **The watchdog** checks `http://127.0.0.1:8888/healthz` every 5 minutes. It
  restarts SearXNG alone (never the stack) on the second consecutive failure,
  after a 10-minute cooldown, like tools-api and laya. It kills every
  `python.exe` whose command line matches `searx\.webapp`, then starts the
  venv's python with `SEARXNG_SETTINGS_PATH` from its `Env` table.
- **`scripts\start-stack.bat`** starts it with `start /B` if the venv exists,
  and skips it otherwise.
- **By hand** (PowerShell), the same thing the watchdog does:

  ```powershell
  Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
      Where-Object { $_.CommandLine -match 'searx\.webapp' } |
      ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
  $env:SEARXNG_SETTINGS_PATH = 'C:\Users\jwals\searxng\etc\settings.yml'
  Start-Process C:\Users\jwals\searxng\.venv\Scripts\python.exe -ArgumentList '-m','searx.webapp' `
      -WorkingDirectory C:\Users\jwals\llama-stack -WindowStyle Hidden `
      -RedirectStandardOutput C:\Users\jwals\llama-stack\logs\searxng.log `
      -RedirectStandardError  C:\Users\jwals\llama-stack\logs\searxng.log.err
  ```

  It answers `/healthz` about 1 s after the start. Restarting it touches
  nothing else in the stack and no GPU.

---

## Failure modes

| what you see | what it is | retryable | remedy |
|---|---|---|---|
| connection refused on :8888 | SearXNG is down | yes, after a restart | the watchdog restarts it on its second strike (up to ~10 min); by hand, see above; `logs\searxng.log.err` says why |
| exits at start with `No module named 'pwd'` | the `.pth` shim is missing (venv rebuilt) | no | recreate `yamadori_windows_shims.pth` (failure 2) |
| exits at start with an `EnvironmentError` about the settings path | `SEARXNG_SETTINGS_PATH` points at nothing | no | run `render_settings.py`; check the launcher's env |
| exits: `server.secret_key is not changed` | settings rendered from a template with the default key, or the env var unset so upstream defaults load | no | render; check `SEARXNG_SETTINGS_PATH` |
| 200, `unresponsive_engines: [["duckduckgo","CAPTCHA"]]` | DDG's bot check; SearXNG suspends the engine for 3,600 s | not for an hour | nothing; other engines carry the query. Bursts cause it |
| `["brave","too many requests"]`, then `Suspended` | Brave's 429; suspended 180 s | after 3 min | the same; space queries out |
| `["npm","timeout"]` (or any engine) on the first query | cold connection setup exceeded 4 s | yes, immediately | none needed |
| 403 on `format=json` | `json` missing from `search.formats` | no | fix the template and re-render |
| docs.rs returns nothing for everything | docs.rs changed its HTML | no | fix the xpaths in the template |
| results for "three.js" about the company Three | bing tokenises `three.js` badly | n/a | its weight is already 0.5; drop it if it keeps leading |

The deep-thinking tool that calls this must turn "connection refused" into the
situation, the fact that it is retryable, and the remedy (the first row), per
"Failure returns carry the next step" in `AGENTS.md`. It must not return an
empty result list.

---

## What leaves the box

Every query goes, as typed, to each engine the request selects: DuckDuckGo,
Brave, Bing, Wikipedia, Stack Exchange and MDN on a plain search, plus
GitHub, npms.io, crates.io and docs.rs with `categories=it`. They are sent
from this machine's public IP with a browser-like user agent, no cookies, and
no account or API key. So the queries are **unattributed but not
anonymous**: an engine sees the IP and the query text, and can link queries
from the same IP. No part of the Yamadori request (conversation, user, key,
repository) is sent, unless the model puts it in the query text. **Code or
paths from a private repo written into a query go out.** The tool that calls
this should say so in its description.

Locally: the werkzeug server writes no access log. When an engine request
fails, the full engine URL, query included, is logged in
`logs\searxng.log.err` (e.g. the brave 429 line). `/stats` keeps
per-engine counters in memory, not queries.

---

## Updating the source

```
cd C:\Users\jwals\searxng
git ls-remote https://github.com/searxng/searxng.git refs/heads/master
curl -sSL -o searxng-master.tar.gz https://github.com/searxng/searxng/archive/refs/heads/master.tar.gz
# move src aside, then:
mkdir src && tar -xzf searxng-master.tar.gz -C src --strip-components=1 --exclude='*:*'
.venv\Scripts\python.exe -m pip install -r src\requirements.txt -r src\requirements-server.txt
.venv\Scripts\python.exe -m pip install --no-build-isolation -e src
```

Then write `src\searx\version_frozen.py` for the new commit, diff
`src\searx\settings.yml` against the old one (we inherit its defaults), check
that `pwd` is still the only POSIX import
(`grep -rn "^import pwd\|^import fcntl\|^import grp\|os.getuid" src/searx`),
restart, and re-run the verification query.

---

## Long term: macOS / Linux

These are supported upstream targets, and none of the Windows workarounds
applies there. `git clone` works, `pwd` exists, and granian can be the
server. A launcher for those hosts should:

- create `searxng/.venv` with a system Python >= 3.10, install from a clone
  (the version comes from git), and render `etc/settings.yml` from the same
  template with the same `render_settings.py`, since the template has no
  paths in it;
- run `granian --interface wsgi --host 127.0.0.1 --port 8888 searx.webapp:app`
  (or keep `python -m searx.webapp` for parity), with
  `SEARXNG_SETTINGS_PATH` set;
- be supervised the same way: `/healthz`, two strikes, restart this service
  alone. It could be a launchd plist / systemd user unit with
  `Restart=on-failure` plus a health timer, or a port of `watchdog.ps1`'s
  service table. Whichever it is, it must kill the whole process group, not
  one PID, because granian forks workers;
- set `YAMADORI_SEARCH_URL` in the proxy's environment the same way.

The upstream container (`src/container/`) is the other route on Linux:
`docker run -p 127.0.0.1:8888:8080` with the rendered settings mounted at
`/etc/searxng/settings.yml`. Bind the host side to 127.0.0.1 explicitly:
Docker's default publishes on every interface.

---

## Not done here

- (Done 2026-09-24, offline: the tool is `search_web` in
  `mcp/research_tools.py`, with the failure return above and the `(web)`
  citation label; see the status at the top.)
- The dashboard vitals (`mcp/vitals.py` `endpoints()`) do not list SearXNG.
- The live suite (`run_tests.py --live`) was not run: nothing in the stack
  uses search yet, and the run would take the GPU.
