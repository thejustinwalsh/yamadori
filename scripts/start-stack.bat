@echo off
setlocal

REM Unattended launcher for the "llama-stack" Scheduled Task.
REM Serves the OpenAI-compatible API AND the dashboard on one port, through
REM the Yamadori proxy (mcp\server.py). llama-swap stays on loopback :11434.
REM
REM   API        http://10.242.120.152:1234/v1
REM   dashboard  http://10.242.120.152:1234/dash   (Python pages: /dash/classic)
REM   tools API  http://10.242.120.152:1235
REM
REM llama-swap starts every model listed in config.yaml via its preload hook.
REM This script also starts tools_api, Caddy (if configured), Laya, the job
REM worker and SearXNG, then runs the proxy in the foreground.

cd /D "%~dp0\.."

REM LOCAL SETTINGS (2026-10-07). scripts\install.ps1 writes stack.env (gitignored) beside this repo's config.yaml:
REM KEY=VALUE lines, `#` comments. A variable already in the environment wins over the file, and the file over
REM the defaults below, so a machine with no stack.env runs exactly as it always has. Keys read here:
REM   YAMADORI_PYTHON      the stack interpreter (default: .venv\Scripts\python.exe in the repo if it exists, else
REM                        %USERPROFILE%\textgen\installer_files\env\python.exe)
REM   YAMADORI_CUDA_BIN    the CUDA runtime DLL directory the engines load (default: the textgen cudabuild env)
REM   YAMADORI_PUBLIC_BASE the URL clients reach this stack at (signed /media links use it)
REM   YAMADORI_SEARXNG_DIR the SearXNG checkout (venv + etc\settings.yml); default %USERPROFILE%\searxng
if exist "stack.env" for /f "usebackq eol=# tokens=1,* delims==" %%A in ("stack.env") do if not defined %%A set "%%A=%%B"

set "SWAP=%CD%\bin\llama-swap.exe"
set "CFG=%CD%\config.yaml"

REM CUDA runtime DLLs for the self-built llama-server live in the isolated
REM conda env created for the toolchain. Without them ggml-cuda.dll fails to
REM load and llama-server exits silently with status 0.
if not defined YAMADORI_CUDA_BIN set "YAMADORI_CUDA_BIN=%USERPROFILE%\textgen\installer_files\cudabuild\Library\bin"
set "PATH=%YAMADORI_CUDA_BIN%;%PATH%"

REM Pin device order so CUDA0/CUDA1 in config.yaml mean what they say.
REM CUDA defaults to fastest-first, which reverses these two cards.
set CUDA_DEVICE_ORDER=PCI_BUS_ID

if not exist "%SWAP%" ( echo ERROR: llama-swap missing at "%SWAP%" & exit /b 1 )
if not exist "%CFG%"  ( echo ERROR: config missing at "%CFG%" & exit /b 1 )

if not exist "logs" mkdir "logs"

REM Docker Engine in the Ubuntu WSL distro (2026-10-06, docs\DOCKER-WSL.md): the MCP host's PackageLens, the npm
REM resolver and the type check run containers there. WSL stops the distro seconds after its last session, so this
REM starts the one hidden keep-alive and waits (60 s at most) for the engine; a failure is logged, never fatal.
REM YAMADORI_WSL_ENGINE=0 (stack.env) skips this on a machine whose Docker is not the WSL engine, or that has none.
if not "%YAMADORI_WSL_ENGINE%"=="0" powershell -NoProfile -ExecutionPolicy Bypass -File "%CD%\scripts\wsl_engine.ps1" -Quiet

REM Code-intelligence HTTP API (port 1235). Same tools as the MCP server,
REM different transport: MCP is for agents, this is for your own software.
REM Every route but /health needs an account key (Authorization: Bearer, the
REM :1234 keys) and refuses a browser Origin it does not list: docs\TOOLS-API.md.
REM Started detached so llama-swap stays this script's foreground process
REM and the Scheduled Task keeps tracking the stack's lifetime correctly.
if not defined YAMADORI_PYTHON if exist "%CD%\.venv\Scripts\python.exe" set "YAMADORI_PYTHON=%CD%\.venv\Scripts\python.exe"
if not defined YAMADORI_PYTHON set "YAMADORI_PYTHON=%USERPROFILE%\textgen\installer_files\env\python.exe"
set "PY=%YAMADORI_PYTHON%"
REM tier models (bench/deploy_tier_models.py)
set "YAMADORI_TIER_MODELS=mcp\tier_models.yaml"
REM max mode (bench/deploy_flash_next.py)
set "YAMADORI_MAX_MODEL=flash-next"
REM layout v3 (bench/deploy_layout_v3.py)
set "YAMADORI_MAIN_CAP=209920"
set "YAMADORI_LANE_TOKENS=0"
set "YAMADORI_SLOTS=1"
REM jjava one-pass reads (operator 2026-10-07; docs/DECIDE-BATCH.md)
set "YAMADORI_DECIDER_BATCH=1"
start "" /B "%PY%" "%CD%\mcp\tools_api.py" >> "logs\tools-api.log" 2>&1

REM TLS reverse proxy. Only started when a DNSimple token is present:
REM without it Caddy cannot complete the ACME DNS-01 challenge, and the
REM stack is still perfectly usable on the raw ports.
if exist "caddy\.env" (
  for /f "usebackq tokens=1,* delims==" %%A in ("caddy\.env") do (
    if /I "%%A"=="DNSIMPLE_API_TOKEN" if not "%%B"=="" set "DNSIMPLE_API_TOKEN=%%B"
  )
)
if defined DNSIMPLE_API_TOKEN (
  echo [%date% %time%] starting caddy >> "logs\caddy.log"
  start "" /B "%CD%\caddy\caddy.exe" run --config "%CD%\caddy\Caddyfile" --adapter caddyfile >> "logs\caddy.log" 2>&1
) else (
  echo [%date% %time%] caddy skipped: no DNSIMPLE_API_TOKEN in caddy\.env >> "logs\caddy.log"
)

REM Laya retired 2026-09-24 (docs/E1.md): E1 heads replace it.

REM Job worker: claims dataset pipeline jobs from index\jobs.sqlite3 (fetch,
REM extract, index). Without it a submitted dataset sits in `queued` forever.
start "" /B "%PY%" "%CD%\mcp\worker.py" >> "logs\worker.log" 2>&1

REM SearXNG web search (loopback 8888, docs/SEARCH.md). Its own venv and
REM source tree outside the repo (YAMADORI_SEARXNG_DIR, default %USERPROFILE%\searxng); the settings file
REM there holds its secret key. Skipped if that venv is missing (nothing in the proxy needs it any more).
if not defined YAMADORI_SEARXNG_DIR set "YAMADORI_SEARXNG_DIR=%USERPROFILE%\searxng"
set "SEARXNG_PY=%YAMADORI_SEARXNG_DIR%\.venv\Scripts\python.exe"
if exist "%SEARXNG_PY%" (
  set "SEARXNG_SETTINGS_PATH=%YAMADORI_SEARXNG_DIR%\etc\settings.yml"
  start "" /B "%SEARXNG_PY%" -m searx.webapp >> "logs\searxng.log" 2>&1
)

echo [%date% %time%] starting llama-swap >> "logs\stack.log"
REM llama-swap binds to LOOPBACK ONLY. It has no authentication of its own, so
REM exposing it is a complete bypass of the proxy: no API key, no tools, no
REM account isolation, no logging. Verified before this change -- a direct
REM request to :1234 with no key returned 200.
REM
REM The proxy takes 1234 instead, which is the port every client is already
REM pointed at, so nothing downstream has to be reconfigured to gain auth.
start "" /B "%SWAP%" -config "%CFG%" -listen 127.0.0.1:11434 >> "logs\stack.log" 2>&1

REM Front door: authenticates, injects tools, detects the repo, logs the corpus.
REM
REM mcp\server.py, NOT mcp\proxy.py. proxy.py holds the request logic and is
REM imported; server.py is the transport. This line still said proxy.py, so an
REM unattended start ran the stdlib BaseHTTPRequestHandler -- no HEAD (health
REM checkers got 501), no real keep-alive, no graceful drain on restart. Every
REM run that worked did so because someone had started server.py by hand.
set "LLAMA_STACK_URL=http://127.0.0.1:11434"
set "YAMADORI_PROXY_PORT=1234"
REM Image generation (docs/IMAGEGEN.md): llama-swap starts `imagegen` on
REM demand; signed /media links use the public name, not 127.0.0.1.
set "YAMADORI_IMAGEGEN_URL=http://127.0.0.1:11434"
if not defined YAMADORI_PUBLIC_BASE set "YAMADORI_PUBLIC_BASE=https://ai.thejustinwalsh.me"
REM Turbo is the default image model (operator, 2026-09-23): 23.6 s vs ~108 s
REM per 1024x1024 (n=1 vs n=10). Quality vs base not yet compared
REM (bench/imagegen/compare_turbo.py). Users can pick base on /settings.
set "YAMADORI_IMAGEGEN_DEFAULT=turbo"
REM Web search for deep thinking: the SearXNG started above (docs/SEARCH.md).
set "YAMADORI_SEARCH_URL=http://127.0.0.1:8888"
REM E1 heads replace Laya (operator, 2026-09-24; docs/E1.md)
set "YAMADORI_E1=1"
"%PY%" "%CD%\mcp\server.py" >> "logs\proxy.log" 2>&1
set EXITCODE=%errorlevel%
echo [%date% %time%] llama-swap exited with %EXITCODE% >> "logs\stack.log"
exit /b %EXITCODE%
