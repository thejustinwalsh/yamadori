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
REM This script also starts tools_api, Caddy (if configured), Laya and the job
REM worker, then runs the proxy in the foreground.

cd /D "%~dp0\.."

set "SWAP=%CD%\bin\llama-swap.exe"
set "CFG=%CD%\config.yaml"

REM CUDA runtime DLLs for the self-built llama-server live in the isolated
REM conda env created for the toolchain. Without them ggml-cuda.dll fails to
REM load and llama-server exits silently with status 0.
set "CUDA_BIN=C:\Users\jwals\textgen\installer_files\cudabuild\Library\bin"
set "PATH=%CUDA_BIN%;%PATH%"

REM Pin device order so CUDA0/CUDA1 in config.yaml mean what they say.
REM CUDA defaults to fastest-first, which reverses these two cards.
set CUDA_DEVICE_ORDER=PCI_BUS_ID

if not exist "%SWAP%" ( echo ERROR: llama-swap missing at "%SWAP%" & exit /b 1 )
if not exist "%CFG%"  ( echo ERROR: config missing at "%CFG%" & exit /b 1 )

if not exist "logs" mkdir "logs"

REM Code-intelligence HTTP API (port 1235). Same tools as the MCP server,
REM different transport: MCP is for agents, this is for your own software.
REM Started detached so llama-swap stays this script's foreground process
REM and the Scheduled Task keeps tracking the stack's lifetime correctly.
set "PY=C:\Users\jwals\textgen\installer_files\env\python.exe"
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

REM Laya decision engine (HTTP 1237 + WebSocket 1238). Runs in its own venv:
REM laya requires a newer transformers than the rest of the stack.
if exist ".venv-laya\Scripts\python.exe" (
  start "" /B ".venv-laya\Scripts\python.exe" "%CD%\mcp\laya_service.py" >> "logs\laya.log" 2>&1
)

REM Job worker: claims dataset pipeline jobs from index\jobs.sqlite3 (fetch,
REM extract, index). Without it a submitted dataset sits in `queued` forever.
start "" /B "%PY%" "%CD%\mcp\worker.py" >> "logs\worker.log" 2>&1

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
"%PY%" "%CD%\mcp\server.py" >> "logs\proxy.log" 2>&1
set EXITCODE=%errorlevel%
echo [%date% %time%] llama-swap exited with %EXITCODE% >> "logs\stack.log"
exit /b %EXITCODE%
