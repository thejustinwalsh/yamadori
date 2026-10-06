@echo off
rem Build bench/fn_probe/h2d_overlap two ways: with CUDA (h2d_overlap.exe, needs the GPU to RUN) and without (h2d_overlap_sim.exe,
rem for --sim: the harness's own test, no GPU, no CUDA call). The staging ring is the engine's ggml-stager.cpp (patch 0030),
rem compiled with GGML_STAGER_NO_BACKEND; STAGER is a source tree that has it (default: the dev tree of 0030; after the
rem candidate is built, STAGER=C:\Users\jwals\engines\llama-upstream-flash-cand0030\src).
rem   usage: build.bat [cuda|sim|both]        env: STAGER, OUT
setlocal
if "%STAGER%"=="" set "STAGER=C:\Users\jwals\engines\dev-stager"
if "%OUT%"=="" set "OUT=C:\Users\jwals\engines\fn_probe"
set "WHAT=%1"
if "%WHAT%"=="" set "WHAT=both"
if not exist "%OUT%" mkdir "%OUT%"
call "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
set "CUDA=C:\Users\jwals\textgen\installer_files\cudabuild\Library"
set "PATH=%CUDA%\bin;%PATH%"
set "HERE=%~dp0"
set "COMMON=-DGGML_STAGER_NO_BACKEND"
if /I not "%WHAT%"=="cuda" (
  cl /nologo /O2 /std:c++17 /EHsc /TP /DPROBE_NO_CUDA %COMMON% /D_CRT_SECURE_NO_WARNINGS /I"%STAGER%\ggml\include" "%HERE%h2d_overlap.cu" "%STAGER%\ggml\src\ggml-stager.cpp" /Fo"%OUT%\\" /Fe:"%OUT%\h2d_overlap_sim.exe" || exit /b 1
)
if /I not "%WHAT%"=="sim" (
  "%CUDA%\bin\nvcc.exe" -O2 -std=c++17 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_120,code=sm_120 %COMMON% -D_CRT_SECURE_NO_WARNINGS -I"%STAGER%\ggml\include" -Xcompiler "/EHsc" "%HERE%h2d_overlap.cu" "%STAGER%\ggml\src\ggml-stager.cpp" -o "%OUT%\h2d_overlap.exe" || exit /b 1
)
echo built into %OUT%
