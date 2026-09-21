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

echo [%date% %time%] starting llama-swap >> "logs\stack.log"
"%SWAP%" -config "%CFG%" -listen 0.0.0.0:1234 >> "logs\stack.log" 2>&1
set EXITCODE=%errorlevel%
echo [%date% %time%] llama-swap exited with %EXITCODE% >> "logs\stack.log"
exit /b %EXITCODE%
