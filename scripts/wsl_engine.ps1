<#
.SYNOPSIS
    Keep Docker Engine (in the Ubuntu WSL distro) reachable for the stack.

.DESCRIPTION
    Yamadori's containers (the MCP host's PackageLens and the npm resolver,
    the type check, the sandbox gate) run on Docker Engine inside the Ubuntu
    WSL distro since 2026-10-06 (docs/DOCKER-WSL.md). WSL terminates a distro
    seconds after its last `wsl.exe` session ends -- systemd services and
    running containers do NOT hold it up (measured 2026-10-06: Ubuntu was
    Stopped within 25 s of the last session, dockerd and a container with it)
    -- so ONE hidden `wsl.exe -d Ubuntu --exec sleep infinity` is the keep-alive.

    Idempotent: a keep-alive that is already running is left alone.

      scripts\wsl_engine.ps1            ensure the keep-alive, wait for the engine to answer
      scripts\wsl_engine.ps1 -Quiet     the same, printing nothing unless it fails (the stack's callers)
      scripts\wsl_engine.ps1 -Status    print what is there; change nothing
      scripts\wsl_engine.ps1 -Stop      end the keep-alive: the VM then shuts down when idle

    Called by scripts\start-stack.bat (before the proxy starts) and by
    scripts\watchdog.ps1 (every run). It runs as the user whose WSL distro it is.
#>
[CmdletBinding()]
param(
    [switch]$Quiet,
    [switch]$Status,
    [switch]$Stop,
    [string]$Distro = 'Ubuntu',
    [int]$WaitSec = 60
)

$ErrorActionPreference = 'Continue'
$root = Split-Path -Parent $PSScriptRoot
$log = Join-Path $root 'logs\wsl-engine.log'
New-Item -ItemType Directory -Force -Path (Split-Path $log) | Out-Null

function Write-Log([string]$msg) {
    $line = "{0}  {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $msg
    Add-Content -Path $log -Value $line
    if (-not $Quiet) { Write-Host $line }
}

function Get-KeepAlive {
    Get-CimInstance Win32_Process -Filter "Name='wsl.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -match 'sleep\s+infinity' }
}

# The engine answers when its API port (127.0.0.1:2376, mutual TLS, WSL's localhost
# forwarding) accepts a connection and `docker version` gets a server back.
function Test-Engine {
    try {
        $c = New-Object Net.Sockets.TcpClient
        $iar = $c.BeginConnect('127.0.0.1', 2376, $null, $null)
        $ok = $iar.AsyncWaitHandle.WaitOne(2000) -and $c.Connected
        $c.Close()
        if (-not $ok) { return $false }
    } catch { return $false }
    $v = & docker version --format '{{.Server.Version}}' 2>$null
    return ($LASTEXITCODE -eq 0 -and [bool]$v)
}

if ($Status) {
    $k = @(Get-KeepAlive)
    "keep-alive processes: {0} ({1})" -f $k.Count, (($k | ForEach-Object { $_.ProcessId }) -join ', ')
    "engine answering:     {0}" -f (Test-Engine)
    "docker context:       {0}" -f (& docker context show)
    exit 0
}

if ($Stop) {
    foreach ($p in @(Get-KeepAlive)) {
        Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
        Write-Log "keep-alive pid $($p.ProcessId) stopped; the VM shuts down when idle"
    }
    exit 0
}

if (-not (Get-KeepAlive)) {
    Start-Process -FilePath 'wsl.exe' -WindowStyle Hidden `
        -ArgumentList @('-d', $Distro, '-u', 'root', '--exec', '/bin/sleep', 'infinity')
    Write-Log "keep-alive started ($Distro)"
}

$t0 = Get-Date
while (-not (Test-Engine)) {
    if (((Get-Date) - $t0).TotalSeconds -ge $WaitSec) {
        Write-Log "FAILED: the engine did not answer on 127.0.0.1:2376 within ${WaitSec}s (docker context $(& docker context show))"
        exit 1
    }
    Start-Sleep -Seconds 2
}
if (-not $Quiet) { Write-Host ("engine up after {0:N1} s" -f ((Get-Date) - $t0).TotalSeconds) }
exit 0
