<#
.SYNOPSIS
    Install Yamadori on a Windows 11 machine: check prerequisites, make the Python environment, write the local
    settings and config.yaml, fetch the pinned models, build the engines, create an account key.

.DESCRIPTION
    Step by step, idempotent (a finished step is recognised and skipped), and never destructive: it does not
    delete anything, does not overwrite stack.env, config.yaml or an existing model file, and every step that
    downloads or compiles asks first (or is pre-approved with -FetchModels / -BuildEngines).

    -DryRun (or -WhatIf) changes nothing at all: it runs the read-only checks and prints what each step WOULD do.
    Run that first.

    What it is built for: two NVIDIA GPUs (the shipped layout is an RTX 5060 Ti 16 GB for the main model and an
    RTX A4000 16 GB for everything else), 64 GB of RAM, Windows 11. docs/INSTALL.md says what is required, what is
    optional, and which numbers in config.example.yaml are measurements of THAT hardware.

.PARAMETER DryRun
    Print what would happen; change nothing.
.PARAMETER ModelsDir
    Where the model files live (default: $HOME\yamadori\models).
.PARAMETER EnginesRoot
    Where llama-server and sd-server are built (default: $HOME\yamadori\engines).
.PARAMETER PublicBase
    The URL clients reach this stack at (default http://127.0.0.1:1234). Signed media links use it.
.PARAMETER Python
    A specific python.exe to make the venv from (default: the newest 3.12+ it finds; 3.13 is what was frozen).
.PARAMETER MainModel
    lean  (default) the main model is sudoingX's published PTQ1_0 + MTP file: one pinned download.
    graft           the exact main model (original trunk + ProCreations' MTP head grafted on, a local build): the
                    file must already exist in ModelsDir; docs/INSTALL.md "The exact main model" says how.
.PARAMETER With
    Optional parts, comma separated: vision, images, mirai (the xhigh tier), flash (the max tier; 64 GB of downloads
    and a manual MTP-draft recipe). Example: -With vision,mirai
.PARAMETER FetchModels
    Pre-approve the model downloads (otherwise each set is confirmed, with its size).
.PARAMETER BuildEngines
    Pre-approve the engine builds (otherwise each is confirmed). Builds use --portable: docs/INSTALL.md.
.PARAMETER Yes
    Answer yes to the questions that are neither a download nor a build.
.PARAMETER SkipAccount
    Do not offer to create an API key.
#>
[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [switch]$DryRun,
    [string]$ModelsDir,
    [string]$EnginesRoot,
    [string]$PublicBase,
    [string]$Python,
    [ValidateSet('lean', 'graft')][string]$MainModel = 'lean',
    [string]$With = '',
    [switch]$FetchModels,
    [switch]$BuildEngines,
    [switch]$Yes,
    [switch]$SkipAccount,
    [string]$AccountLabel = 'me'
)

$ErrorActionPreference = 'Stop'
try { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12 } catch { }
$Repo = Split-Path -Parent $PSScriptRoot
$Dry = [bool]($DryRun -or $WhatIfPreference)
# -With takes a comma list (a plain string, because `powershell -File` does not split `a,b` into an array)
$WithList = @($With -split '[,\s]+' | Where-Object { $_ })
foreach ($w in $WithList) { if (@('vision', 'images', 'mirai', 'flash') -notcontains $w) { throw "-With: '$w' is not one of vision, images, mirai, flash" } }
$script:Done = New-Object System.Collections.ArrayList
$script:Todo = New-Object System.Collections.ArrayList
$script:Warned = New-Object System.Collections.ArrayList

# ---------------------------------------------------------------------------------------------------- the plan
# The model sets. Id = models/manifest.yaml artifact id (scripts/fetch_models.py fetches it); GiB = its size there.
# mcp/test_install_docs.py checks every Id exists in the manifest with this size.
$ModelSets = [ordered]@{
    core   = @{ Title = 'Bonsai tiers (minimal .. high): the trunk for the second card, the embedder'; Items = @(
        @{ Id = 'bonsai-2-27b-original';       GiB = 5.54; Why = 'Bonsai 2 27B PTQ1_0 trunk: bonsai-a4000 (jjava, side calls) serves it' }
        @{ Id = 'qwen3-embedding-0.6b-q8';     GiB = 0.60; Why = 'the embedder (code search, the skills)' } ) }
    lean   = @{ Title = 'main model, quick path: PTQ1_0 + MTP head in one published file'; Items = @(
        @{ Id = 'bonsai-2-27b-mtp-lean-stock'; GiB = 5.87; Why = 'the main model (a pinned download; expected to run, not measured by us as the main model)' } ) }
    vision = @{ Title = 'looking at images'; Items = @(
        @{ Id = 'bonsai-2-27b-mmproj-q8';      GiB = 0.59; Why = 'vision projector for bonsai-vision (second card, on demand)' } ) }
    images = @{ Title = 'drawing (Qwen Research License: non-commercial)'; Items = @(
        @{ Id = 'qwen-image-2.1-q5km';         GiB = 5.02; Why = 'image denoiser, base (20 steps)' }
        @{ Id = 'qwen-image-2.1-turbo-q5km';   GiB = 4.66; Why = 'image denoiser, turbo (4 steps)' }
        @{ Id = 'qwen-image-2.1-text-encoder-heretic-q4km'; GiB = 4.68; Why = 'image text encoder' }
        @{ Id = 'qwen-image-2.1-vae-bf16';     GiB = 0.63; Why = 'image VAE' } ) }
    mirai  = @{ Title = 'the xhigh tier: Mirai S Qwen3.8-27B'; Items = @(
        @{ Id = 'mirai-s-qwen3.8-27b-gguf';    GiB = 10.41; Why = 'Mirai S 2.4-bit GGUF (a third-party conversion; needs the llama-mirai-s engine)' } ) }
    flash  = @{ Title = 'the max tier: Qwen3.8-Flash-Next IQ2_XS (needs ~38 GB of RAM beside the card)'; Items = @(
        @{ Id = 'flash-next-iq2xs-shard1';     GiB = 36.53; Why = 'Flash-Next weights, shard 1' }
        @{ Id = 'flash-next-iq2xs-shard2-ngram'; GiB = 26.82; Why = 'Flash-Next n-gram table, shard 2' } ) }
}
# Engines (engines/manifest.yaml): name, what for, needed for which set
$Engines = @(
    @{ Name = 'llama-bonsai2-ada';    For = 'bonsai, bonsai-a4000 and (here) the embedder and bonsai-vision'; Set = 'core' }
    @{ Name = 'llama-mirai-s';        For = 'mirai-s (the xhigh tier)';       Set = 'mirai' }
    @{ Name = 'llama-upstream-flash'; For = 'flash-next (the max tier)';      Set = 'flash' }
    @{ Name = 'sd-cpp';               For = 'imagegen, imagegen-turbo';       Set = 'images' }
)

# ------------------------------------------------------------------------------------------------- the helpers
function Say([string]$m)  { Write-Host $m }
function Step([int]$n, [string]$title) { Write-Host ''; Write-Host ("[{0}/10] {1}" -f $n, $title) -ForegroundColor Cyan }
function Ok([string]$m)   { Write-Host "  ok    $m" -ForegroundColor Green }
function Warn([string]$m) { Write-Host "  WARN  $m" -ForegroundColor Yellow; [void]$script:Warned.Add($m) }
function Info([string]$m) { Write-Host "        $m" }
function Would([string]$m){ Write-Host "  DRY   would $m" -ForegroundColor DarkGray }
function Todo([string]$m) { Write-Host "  TODO  $m" -ForegroundColor Magenta; [void]$script:Todo.Add($m) }
function Done([string]$m) { [void]$script:Done.Add($m) }

function Ask([string]$question, [switch]$Download) {
    # A download or a build is never answered by -Yes: only by its own switch or by a person.
    if ($Dry) { Would "ask: $question"; return $false }
    if ($Yes -and -not $Download) { return $true }
    $a = Read-Host "  ??    $question [y/N]"
    return ($a -match '^(y|yes)$')
}

function Invoke-Native([string]$exe, [string[]]$arguments) {
    # Run a program, stream its output, return the exit code. Never throws on a non-zero exit.
    & $exe @arguments
    return $LASTEXITCODE
}

function Get-ExeOutput([string]$exe, [string[]]$arguments) {
    try { $o = & $exe @arguments 2>$null; if ($LASTEXITCODE -eq 0) { return ($o | Out-String).Trim() } } catch { }
    return $null
}

function Resolve-Python {
    $cands = @()
    if ($Python) { $cands += , @($Python) }
    $cands += , @('py', '-3.13'), @('py', '-3.12'), @('python'), @('python3')
    foreach ($c in $cands) {
        $exe = $c[0]; $rest = @(); if ($c.Count -gt 1) { $rest = $c[1..($c.Count - 1)] }
        if (-not (Get-Command $exe -ErrorAction SilentlyContinue)) { continue }
        $v = Get-ExeOutput $exe ($rest + @('-c', "import sys; print('%d.%d.%d' % sys.version_info[:3]); print(sys.executable)"))
        if (-not $v) { continue }
        $lines = $v -split "`r?`n"
        if ($lines.Count -lt 2 -or $lines[0] -notmatch '^(\d+)\.(\d+)\.(\d+)$') { continue }
        if ([int]$Matches[1] -eq 3 -and [int]$Matches[2] -ge 12) { return @{ Version = $lines[0]; Exe = $lines[1] } }
    }
    return $null
}

function Get-Gpus {
    $o = Get-ExeOutput 'nvidia-smi' @('--query-gpu=index,name,memory.total,uuid,driver_version', '--format=csv,noheader,nounits')
    $gpus = @()
    if ($o) {
        foreach ($l in ($o -split "`r?`n")) {
            $p = $l.Split(',') | ForEach-Object { $_.Trim() }
            if ($p.Count -ge 5) { $gpus += @{ Index = [int]$p[0]; Name = $p[1]; MiB = [int]$p[2]; Uuid = $p[3]; Driver = $p[4] } }
        }
    }
    return $gpus
}

function Get-FreeGiB([string]$path) {
    try {
        $full = [System.IO.Path]::GetFullPath($path)
        $drive = (Get-PSDrive -Name $full.Substring(0, 1) -ErrorAction Stop)
        return [math]::Round($drive.Free / 1GB, 1)
    } catch { return $null }
}

function Get-ModelPath([string]$id) {
    # ${models}-relative path of a manifest artifact, read from models/manifest.yaml (a line scan; the ids are unique)
    $lines = Get-Content (Join-Path $Repo 'models\manifest.yaml')
    for ($i = 0; $i -lt $lines.Count; $i++) {
        if ($lines[$i] -match ('^\s+- id: ' + [regex]::Escape($id) + '\s*$')) {
            for ($j = $i + 1; $j -lt [math]::Min($i + 12, $lines.Count); $j++) {
                if ($lines[$j] -match '^\s+path:\s*"\$\{models\}/(.+)"\s*$') { return $Matches[1] }
            }
        }
    }
    return $null
}

function Get-SelectedSets {
    $sets = @('core')
    if ($MainModel -eq 'lean') { $sets += 'lean' }
    foreach ($w in $WithList) { $sets += $w }
    return $sets
}

# ----------------------------------------------------------------------------------------------------- start
Say ''
Say 'Yamadori installer'
Say ("  repository   {0}" -f $Repo)
if ($Dry) { Say '  DRY RUN      nothing will be changed; the read-only checks run and each step says what it would do' }
if (-not $ModelsDir)   { $ModelsDir = Join-Path $HOME 'yamadori\models' }
if (-not $EnginesRoot) { $EnginesRoot = Join-Path $HOME 'yamadori\engines' }
if (-not $PublicBase)  { $PublicBase = 'http://127.0.0.1:1234' }
$ModelsDir = [System.IO.Path]::GetFullPath($ModelsDir)
$EnginesRoot = [System.IO.Path]::GetFullPath($EnginesRoot)
Say ("  models       {0}" -f $ModelsDir)
Say ("  engines      {0}" -f $EnginesRoot)
Say ("  clients      {0}   (-PublicBase changes it)" -f $PublicBase)
Say ("  main model   {0}" -f $MainModel)
Say ("  optional     {0}" -f $(if ($WithList.Count) { $WithList -join ', ' } else { 'none (add with -With vision,images,mirai,flash)' }))

if ($env:OS -ne 'Windows_NT') { throw 'This installer is for Windows 11 (the start scripts are .bat and .ps1). docs/INSTALL.md.' }

# ------------------------------------------------------------------------------------ 1. prerequisites
Step 1 'Prerequisites (read-only)'
$gpus = @(Get-Gpus)
if ($gpus.Count -eq 0) {
    Warn 'nvidia-smi lists no GPU. Install the NVIDIA driver (a current Game Ready / Studio driver) and run this again.'
} else {
    foreach ($g in $gpus) { Ok ("GPU {0}: {1}, {2:N0} MiB, driver {3}" -f $g.Index, $g.Name, $g.MiB, $g.Driver) }
    if ($gpus.Count -lt 2) {
        Warn 'One GPU. The shipped layout needs two (main model alone on one card; jjava, side calls, images, embedder on the other). A single-GPU layout is not provided: docs/INSTALL.md "Hardware".'
    }
}
$ram = $null
try { $ram = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB) } catch { }
if ($ram) {
    if ($ram -ge 64) { Ok "RAM $ram GB" }
    elseif ($WithList -contains 'flash') { Warn "RAM $ram GB; Flash-Next (-With flash) keeps ~38 GB of expert weights in RAM and the box was built with 64 GB." }
    else { Ok "RAM $ram GB (64 GB is what it was built with; the Bonsai tiers use far less)" }
}
$py = Resolve-Python
if ($py) { Ok ("Python {0} at {1}" -f $py.Version, $py.Exe) }
else { Warn 'No Python 3.12+ found. Install Python 3.13 from python.org (tick "Add python.exe to PATH") or pass -Python <path>.' }
if (Get-Command git -ErrorAction SilentlyContinue) { Ok ((Get-ExeOutput 'git' @('--version'))) } else { Warn 'git not found (needed to build engines and to fetch the main-model graft inputs). Install Git for Windows.' }

$vsPath = $null
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
if (Test-Path $vswhere) {
    $vsPath = Get-ExeOutput $vswhere @('-latest', '-products', '*', '-requires', 'Microsoft.VisualStudio.Component.VC.Tools.x86.x64', '-property', 'installationPath')
}
$cudaRoot = $env:CUDA_PATH
$nvcc = Get-Command nvcc -ErrorAction SilentlyContinue
if ($vsPath) { Ok "Visual Studio with the C++ tools at $vsPath (engine builds)" } else { Info 'Visual Studio 2022 with "Desktop development with C++": only needed to BUILD engines.' }
if ($cudaRoot -and (Test-Path (Join-Path $cudaRoot 'bin'))) { Ok "CUDA toolkit at $cudaRoot (engine builds; its bin\ is also what the servers load)" }
elseif ($nvcc) { Ok "nvcc at $($nvcc.Source)" }
else { Info 'CUDA toolkit 12.8 (12.x): only needed to BUILD engines; the recorded toolchain used 12.8.93.' }

$dockerVer = Get-ExeOutput 'docker' @('version', '--format', '{{.Server.Version}}')
if ($dockerVer) { Ok "Docker engine $dockerVer answers (optional: the package lookups run PackageLens in it)" }
else { Info 'No Docker engine answers (optional: only the package lookups, yama_find_package and friends, need one). docs/DOCKER-WSL.md is how it was set up here.' }

$sets = Get-SelectedSets
$needGiB = 0.0
foreach ($s in $sets) { foreach ($it in $ModelSets[$s].Items) { $needGiB += $it.GiB } }
$free = Get-FreeGiB $ModelsDir
if ($free -ne $null) {
    if ($free -gt ($needGiB + 5)) { Ok ("disk: {0} GiB free where the models go; the selected sets are {1:N1} GiB" -f $free, $needGiB) }
    else { Warn ("disk: {0} GiB free where the models go; the selected sets are {1:N1} GiB" -f $free, $needGiB) }
}

# ------------------------------------------------------------------------------------------ 2. venv
Step 2 'Python environment (.venv from requirements.txt)'
$VenvPy = Join-Path $Repo '.venv\Scripts\python.exe'
if (Test-Path $VenvPy) { Ok ".venv exists ($VenvPy)" }
elseif (-not $py) { Todo 'make .venv once a Python 3.12+ is installed' }
elseif ($Dry) { Would ("create .venv with {0} and run: pip install -r requirements.txt (about 15 packages and their dependencies, a few hundred MB from PyPI)" -f $py.Exe) }
elseif (Ask 'Create .venv and install requirements.txt from PyPI (a few hundred MB)?') {
    [void](Invoke-Native $py.Exe @('-m', 'venv', (Join-Path $Repo '.venv')))
}
if ((Test-Path $VenvPy) -and -not $Dry) {
    $have = Get-ExeOutput $VenvPy @('-c', 'import fastapi, uvicorn, numpy, yaml, psutil, PIL, tree_sitter_language_pack, markdown_it, websockets; print("ok")')
    if ($have -eq 'ok') { Ok 'the packages are installed' }
    elseif (Ask 'Install the packages into .venv now (pip, from PyPI)?') {
        $rc = Invoke-Native $VenvPy @('-m', 'pip', 'install', '--disable-pip-version-check', '-r', (Join-Path $Repo 'requirements.txt'))
        if ($rc -eq 0) { Ok 'packages installed'; Done 'Python environment' } else { Warn "pip exited $rc" }
    } else { Todo '.venv\Scripts\python.exe -m pip install -r requirements.txt' }
} elseif ((Test-Path $VenvPy) -and $Dry) { Would 'check the packages import, and pip install -r requirements.txt if not' }
$StackPy = $VenvPy

# ------------------------------------------------------------------------------------- 3. stack.env
Step 3 'stack.env (this machine''s interpreter, CUDA runtime and URL; gitignored)'
$StackEnv = Join-Path $Repo 'stack.env'
if (Test-Path $StackEnv) { Ok 'stack.env exists; left alone' }
else {
    $cudaBin = if ($cudaRoot) { Join-Path $cudaRoot 'bin' } else { '' }
    $lines = @(
        "# Written by scripts/install.ps1 on $(Get-Date -Format 'yyyy-MM-dd'). KEY=VALUE, '#' comments. Read by scripts/start-stack.bat and",
        '# scripts/watchdog.ps1; a variable already in the environment wins over this file. Not committed.',
        "YAMADORI_PYTHON=$VenvPy",
        "YAMADORI_PUBLIC_BASE=$PublicBase",
        "YAMADORI_MODELS_DIR=$ModelsDir"
    )
    if ($cudaBin) { $lines += "YAMADORI_CUDA_BIN=$cudaBin" }
    else { $lines += '# YAMADORI_CUDA_BIN=<a directory holding the CUDA 12 runtime DLLs (cudart64_12.dll, cublas64_12.dll) the engines load>' }
    if (-not $dockerVer) {
        $lines += '# No Docker engine was found: the package lookups (PackageLens) are switched off and the WSL keep-alive is skipped.'
        $lines += 'YAMADORI_MCP_TOOLS=0'
        $lines += 'YAMADORI_WSL_ENGINE=0'
    }
    if ($Dry) { Would ("write {0}:" -f $StackEnv); $lines | ForEach-Object { Info $_ } }
    elseif (Ask 'Write stack.env?') { Set-Content -Path $StackEnv -Value $lines -Encoding ascii; Ok 'wrote stack.env'; Done 'stack.env' }
    else { Todo 'write stack.env (see docs/INSTALL.md "stack.env")' }
}

# --------------------------------------------------------------------------------------- 4. llama-swap
Step 4 'llama-swap (the router in front of every model server; a pinned release binary)'
$Swap = Join-Path $Repo 'bin\llama-swap.exe'
if (Test-Path $Swap) { Ok "bin\llama-swap.exe present ($((Get-ExeOutput $Swap @('--version')) -replace "`r?`n", ' '))" }
else {
    $mf = Get-Content (Join-Path $Repo 'engines\manifest.yaml') -Raw
    if ($mf -match '(?s)\n  llama-swap:.*?asset:\s*\n\s+url:\s*(\S+)\s*\n\s+sha256:\s*([0-9a-f]{64})') {
        $swapUrl = $Matches[1]; $swapSha = $Matches[2]
        if ($Dry) { Would "download $swapUrl, check sha256 $swapSha, extract llama-swap.exe to bin\" }
        elseif (Ask "Download $swapUrl (a few MB) and verify its sha256?" -Download) {
            New-Item -ItemType Directory -Force -Path (Join-Path $Repo 'bin') | Out-Null
            $zip = Join-Path $env:TEMP 'llama-swap.zip'
            Invoke-WebRequest -Uri $swapUrl -OutFile $zip -UseBasicParsing
            $got = (Get-FileHash $zip -Algorithm SHA256).Hash.ToLower()
            if ($got -ne $swapSha) { Remove-Item $zip; throw "llama-swap zip sha256 $got is not the pinned $swapSha" }
            Expand-Archive -Path $zip -DestinationPath (Join-Path $Repo 'bin') -Force
            Remove-Item $zip
            Ok 'bin\llama-swap.exe installed (sha256 of the release verified)'; Done 'llama-swap'
        } else { Todo "download llama-swap: $swapUrl into bin\ (engines/manifest.yaml pins its sha256)" }
    } else { Warn 'could not read the llama-swap pin from engines/manifest.yaml' }
}

# ---------------------------------------------------------------------------------------- 5. engines
Step 5 'Engines (llama-server and sd-server, built from the source vendored in engines/src)'
$buildNeeded = @($Engines | Where-Object { $sets -contains $_.Set })
$toolLocal = Join-Path $Repo 'engines\manifest.local.yaml'
foreach ($e in $buildNeeded) {
    $exe = if ($e.Name -eq 'sd-cpp') { 'sd-server.exe' } else { 'llama-server.exe' }
    $built = Join-Path $EnginesRoot "$($e.Name)\src\build\bin\$exe"
    if (Test-Path $built) { Ok "$($e.Name): $built"; continue }
    $cmd = "$StackPy scripts\build_engine.py build $($e.Name) --portable --no-tests --jobs $([Environment]::ProcessorCount) --out $(Join-Path $EnginesRoot $e.Name)"
    Info "$($e.Name) for $($e.For)"
    if (-not $vsPath -or -not ($cudaRoot -or $nvcc)) {
        Todo "build $($e.Name) once Visual Studio 2022 (C++ tools) and the CUDA 12 toolkit are installed: $cmd"
        continue
    }
    if ($Dry) { Would "build: $cmd   (tens of minutes; CPU and disk heavy)"; continue }
    if (-not (Test-Path $toolLocal)) {
        Info 'engines\manifest.local.yaml tells the builder where THIS machine''s Visual Studio, CUDA and Git are.'
        $gitExe = (Get-Command git -ErrorAction SilentlyContinue).Source
        $gitRoot = if ($gitExe) { Split-Path (Split-Path $gitExe -Parent) -Parent } else { '' }
        $cm = Join-Path $vsPath 'Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe'
        $nj = Join-Path $vsPath 'Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe'
        $y = @(
            '# Written by scripts/install.ps1. Merged over engines/manifest.yaml `defaults` and `toolchains` only (scripts/build_engine.py).',
            'defaults:', ("  engines_root: " + ($EnginesRoot -replace '\\', '/')),
            'toolchains:', '  msvc-cuda128:',
            ("    vcvars: " + ((Join-Path $vsPath 'VC\Auxiliary\Build\vcvars64.bat') -replace '\\', '/')),
            '    cmake:', ("      path: " + ($cm -replace '\\', '/')),
            '    ninja:', ("      path: " + ($nj -replace '\\', '/')),
            '    cuda:', ("      root: " + (($(if ($cudaRoot) { $cudaRoot } else { Split-Path (Split-Path $nvcc.Source -Parent) -Parent })) -replace '\\', '/')),
            '    git:', ("      path: " + ($gitExe -replace '\\', '/')),
            '    gzip:', ("      path: " + ((Join-Path $gitRoot 'usr\bin\gzip.exe') -replace '\\', '/'))
        )
        if (Ask 'Write engines\manifest.local.yaml from what was detected?') { Set-Content -Path $toolLocal -Value $y -Encoding ascii; Ok 'wrote engines\manifest.local.yaml' }
        else { Todo 'write engines\manifest.local.yaml (docs/INSTALL.md "Building the engines")'; continue }
    }
    if ($BuildEngines -or (Ask "Build $($e.Name) now? It compiles CUDA code: tens of minutes, several GB of disk." -Download)) {
        $rc = Invoke-Native $StackPy @('scripts\build_engine.py', 'build', $e.Name, '--portable', '--no-tests', '--jobs', "$([Environment]::ProcessorCount)", '--out', (Join-Path $EnginesRoot $e.Name))
        if ($rc -eq 0) { Ok "built $($e.Name)"; Done "engine $($e.Name)" } else { Warn "build_engine.py exited $rc for $($e.Name); its log is in $(Join-Path $EnginesRoot $e.Name)" }
    } else { Todo "build $($e.Name): $cmd" }
}
if ($sets -contains 'flash') {
    Todo 'Flash-Next also needs its MTP draft file and the expert profile, which are recipes, not downloads: models/manifest.yaml flash-next-mtp-draft-q8_0 and flash-next-expert-profile-strata (docs/INSTALL.md).'
}

# ----------------------------------------------------------------------------------------- 6. models
Step 6 'Models (pinned Hugging Face revisions, sha256-verified by scripts/fetch_models.py)'
$total = 0.0
foreach ($s in $sets) {
    Info ("{0}: {1}" -f $s, $ModelSets[$s].Title)
    foreach ($it in $ModelSets[$s].Items) {
        $rel = Get-ModelPath $it.Id
        $target = if ($rel) { Join-Path $ModelsDir ($rel -replace '/', '\') } else { $null }
        if ($target -and (Test-Path $target)) { Ok ("{0,-46} {1,6:N2} GiB  present" -f $it.Id, $it.GiB); continue }
        $total += $it.GiB
        Info ("{0,-46} {1,6:N2} GiB  {2}" -f $it.Id, $it.GiB, $it.Why)
    }
}
if ($MainModel -eq 'graft') {
    $g = Join-Path $ModelsDir 'Ternary-Bonsai-2-27B-PTQ1_0-mtp-procreations.gguf'
    if (Test-Path $g) { Ok 'the exact main model (graft) is present' }
    else { Todo 'build the exact main model: docs/INSTALL.md "The exact main model" (a local graft, ~10 GB of ranged downloads and one script run)' }
}
if ($total -gt 0) {
    Say ("        to download: {0:N1} GiB from huggingface.co" -f $total)
    if ($Dry) {
        foreach ($s in $sets) {
            foreach ($it in $ModelSets[$s].Items) {
                $rel = Get-ModelPath $it.Id
                if ($rel -and (Test-Path (Join-Path $ModelsDir ($rel -replace '/', '\')))) { continue }
                Would ("run: {0} scripts\fetch_models.py fetch {1} --models-dir {2}" -f $StackPy, $it.Id, $ModelsDir)
            }
        }
    }
    elseif ($FetchModels -or (Ask ("Download {0:N1} GiB of model files now?" -f $total) -Download)) {
        if (-not (Test-Path $VenvPy)) { Todo 'make the .venv first (step 2), then run this again' }
        else {
            foreach ($s in $sets) {
                foreach ($it in $ModelSets[$s].Items) {
                    $rel = Get-ModelPath $it.Id
                    if ($rel -and (Test-Path (Join-Path $ModelsDir ($rel -replace '/', '\')))) { continue }
                    $rc = Invoke-Native $StackPy @('scripts\fetch_models.py', 'fetch', $it.Id, '--models-dir', $ModelsDir)
                    if ($rc -ne 0) { Warn "fetching $($it.Id) exited $rc" }
                }
            }
            Done 'models'
        }
    } else { Todo ("download the models: {0} scripts\fetch_models.py fetch <id> --models-dir {1}  (ids above)" -f $StackPy, $ModelsDir) }
}

# --------------------------------------------------------------------------------------- 7. config.yaml
Step 7 'config.yaml (llama-swap''s config, rendered from config.example.yaml)'
$Cfg = Join-Path $Repo 'config.yaml'
if (Test-Path $Cfg) { Ok 'config.yaml exists; left alone (python scripts\make_config.py --force re-renders it, keeping a .bak copy)' }
else {
    $mainFile = if ($MainModel -eq 'lean') { 'Ternary-Bonsai-2-27B-PTQ1_0-mtp-lean.gguf' } else { 'Ternary-Bonsai-2-27B-PTQ1_0-mtp-procreations.gguf' }
    $ada = (Join-Path $EnginesRoot 'llama-bonsai2-ada\src\build\bin\llama-server.exe') -replace '\\', '/'
    $mk = @('scripts\make_config.py', '--models-dir', $ModelsDir, '--engines-root', $EnginesRoot,
            '--set', "MAIN_MODEL=$mainFile", '--set', "SERVER_PRISM=$ada")
    if ($gpus.Count -ge 2) {
        $main = $gpus[0]; $second = $gpus[1]
        Info ("main card   GPU {0}: {1}" -f $main.Index, $main.Name)
        Info ("second card GPU {0}: {1}" -f $second.Index, $second.Name)
        Info 'nvidia-smi lists cards in PCI bus order, as start-stack.bat pins (CUDA_DEVICE_ORDER=PCI_BUS_ID). The MAIN card is the one the 27B lives alone on.'
        $mk += @('--main-gpu', $main.Uuid, '--second-gpu', $second.Uuid)
    } else { Warn 'fewer than two GPUs: config.yaml cannot be rendered with both card UUIDs; pass --main-gpu and --second-gpu to scripts\make_config.py yourself' }
    if ($gpus.Count -ge 2) {
        if ($Dry) { Would ("run: {0} {1}" -f $StackPy, ($mk -join ' ')) }
        elseif ((Test-Path $VenvPy) -and (Ask 'Write config.yaml with these cards?')) {
            [void](Invoke-Native $StackPy $mk); Done 'config.yaml'
        } else { Todo ("render config.yaml: {0} {1}" -f $StackPy, ($mk -join ' ')) }
    }
}

# ------------------------------------------------------------------------------------ 8. token embeddings
Step 8 'Concept-seed vocabulary (index/token_embd.npz, from the 27B''s own token embeddings; CPU, about a minute)'
$Npz = Join-Path $Repo 'index\token_embd.npz'
$trunk = Join-Path $ModelsDir 'Ternary-Bonsai-2-27B-PTQ1_0.gguf'
if (Test-Path $Npz) { Ok 'index\token_embd.npz exists' }
elseif (-not (Test-Path $trunk)) { Todo "extract it once the trunk is downloaded: $StackPy scripts\extract_token_embd.py --gguf $trunk" }
elseif ($Dry) { Would "run: $StackPy scripts\extract_token_embd.py --gguf $trunk  (writes a 420 MB index\token_embd.npz)" }
elseif (Ask 'Extract it now (reads the 5.5 GB trunk, writes a 420 MB file)?') {
    [void](Invoke-Native $StackPy @('scripts\extract_token_embd.py', '--gguf', $trunk)); Done 'token embeddings'
}
if (-not (Test-Path $Npz)) { Info 'Without it the concept seed (tiers low and up) cannot draw a word; the rest of the stack runs.' }

# ------------------------------------------------------------------------------------------ 9. account
Step 9 'An API key (until one account exists the server is open; with one, every request needs its key)'
$Accts = Join-Path $Repo 'index\accounts\accounts.json'
if ($SkipAccount) { Info 'skipped (-SkipAccount)' }
elseif (Test-Path $Accts) { Ok 'an account registry exists' }
elseif ($Dry) { Would "run: $StackPy mcp\accounts.py create $AccountLabel   and save the key to .keys\$AccountLabel.key (gitignored)" }
elseif (-not (Test-Path $VenvPy)) { Todo "create a key: $StackPy mcp\accounts.py create $AccountLabel" }
elseif (Ask "Create an API key for '$AccountLabel'? It is saved to .keys\$AccountLabel.key; only its hash is stored by the server.") {
    $out = & $StackPy (Join-Path $Repo 'mcp\accounts.py') create $AccountLabel
    $key = ($out | Where-Object { $_ -match '^\s+\S{16,}\s*$' } | Select-Object -Last 1)
    if ($key) {
        New-Item -ItemType Directory -Force -Path (Join-Path $Repo '.keys') | Out-Null
        Set-Content -Path (Join-Path $Repo ".keys\$AccountLabel.key") -Value $key.Trim() -Encoding ascii
        Ok "key saved to .keys\$AccountLabel.key (clients send it as Authorization: Bearer <key>)"; Done 'account key'
    } else { Warn 'could not read the key from the output of mcp\accounts.py; run it yourself and keep what it prints' }
}

# ------------------------------------------------------------------------------------ 10. docker/PackageLens
Step 10 'Package lookups (optional): PackageLens in a container'
if (-not $dockerVer) { Info 'no Docker engine answers: skipped (stack.env switched the lookups off). Later: docs/DOCKER-WSL.md, then remove YAMADORI_MCP_TOOLS=0.' }
else {
    $img = Get-ExeOutput 'docker' @('image', 'inspect', '--format', '{{.Id}}', 'yamadori-mcp-packagelens:0.1.11')
    if ($img) { Ok "image yamadori-mcp-packagelens:0.1.11 is built ($img)" }
    else {
        $df = Get-Content (Join-Path $Repo 'mcp_servers\packagelens\Dockerfile') -Raw
        $base = if ($df -match '(?m)^ARG BASE=(\S+)') { $Matches[1] } else { '<the ARG BASE of mcp_servers\packagelens\Dockerfile>' }
        if ($Dry) { Would "docker pull $base ; $StackPy mcp\mcp_host.py build packagelens ; then record the printed image id in models/manifest.yaml (mcp-packagelens image_id)" }
        else {
            Todo "docker pull $base   (a download), then: $StackPy mcp\mcp_host.py build packagelens"
            Todo 'a rebuild gets a NEW image id (npm ci writes timestamps); the proxy starts the server only from the id models/manifest.yaml records, so put the id the build prints into runtimes: mcp-packagelens `image_id` (docs/INSTALL.md)'
        }
    }
}

# ---------------------------------------------------------------------------------------- summary
Write-Host ''
Write-Host 'Summary' -ForegroundColor Cyan
if ($Dry) { Say '  DRY RUN: nothing was changed.' }
foreach ($d in @($script:Done)) { Write-Host "  done  $d" -ForegroundColor Green }
foreach ($t in @($script:Todo)) { Write-Host "  TODO  $t" -ForegroundColor Magenta }
foreach ($w in @($script:Warned)) { Write-Host "  WARN  $w" -ForegroundColor Yellow }
if (-not $script:Done.Count -and -not $script:Todo.Count -and -not $script:Warned.Count -and -not $Dry) { Say '  nothing to do: everything was already in place.' }
Say ''
Say 'Start it:'
Say '  scripts\start-stack.bat                  (a console: logs go to logs\*.log; Ctrl+C stops it)'
Say '  scripts\install-autostart.ps1            (start at logon + a watchdog every 5 minutes; -DryRun first)'
Say 'Then:'
Say ("  curl.exe {0}/health" -f $PublicBase)
Say ("  open {0}/   (the dashboard)" -f $PublicBase)
Say ('  point any OpenAI client at {0}/v1 with the key from .keys\ (model name: yamadori; reasoning_effort picks the tier)' -f $PublicBase)
Say '  .venv\Scripts\python.exe scripts\run_tests.py       (the offline suites)'
Say '  docs/INSTALL.md says what is required, what is optional, and what this installer cannot do for you.'
