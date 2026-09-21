@echo off
setlocal

REM Unattended launcher for the "llama-stack" Scheduled Task.
REM Serves the OpenAI-compatible API AND the control/stats UI on one port.
REM
REM   API   http://10.242.120.152:1234/v1
REM   UI    http://10.242.120.152:1234/ui/
REM
REM llama-swap starts every model listed in config.yaml via its preload hook,
REM so there is nothing else to launch.

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

echo [%date% %time%] starting llama-swap >> "logs\stack.log"
"%SWAP%" -config "%CFG%" -listen 0.0.0.0:1234 >> "logs\stack.log" 2>&1
set EXITCODE=%errorlevel%
echo [%date% %time%] llama-swap exited with %EXITCODE% >> "logs\stack.log"
exit /b %EXITCODE%
