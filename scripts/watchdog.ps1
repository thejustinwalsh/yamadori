<#
.SYNOPSIS
    Health check and self-heal for the llama-stack.

.DESCRIPTION
    Runs on a schedule. Verifies, in increasing order of strictness:

      1. llama-swap process is alive
      2. the API answers on :1234
      3. every model in the resident group reports "loaded"
      4. the primary actually generates a token

    Any failure triggers a restart of the "llama-stack" Scheduled Task.
    A restart is never attempted more than once per RestartCooldownMin, so a
    model that crashes on load cannot become a restart loop -- it fails loudly
    in the log instead of thrashing the GPUs.

    Deliberately does NOT restart when an on-demand model (vision) is loaded
    and the resident set is temporarily evicted: that is normal operation, not
    a fault.
#>
[CmdletBinding()]
param(
    [string]$Api = 'http://127.0.0.1:1234',
    [string]$ToolsApi = 'http://127.0.0.1:1235',
    [string]$LayaApi  = 'http://127.0.0.1:1237',
    [string]$TaskName = 'llama-stack',
    [string[]]$Resident = @('bonsai', 'embeddings', 'reranker'),
    [int]$RestartCooldownMin = 10,
    [switch]$WhatIfOnly
)

$ErrorActionPreference = 'Continue'
$root = Split-Path -Parent $PSScriptRoot
$log = Join-Path $root 'logs\watchdog.log'
$stamp = Join-Path $root 'logs\.last-restart'
New-Item -ItemType Directory -Force -Path (Split-Path $log) | Out-Null

function Write-Log([string]$msg) {
    $line = "{0}  {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $msg
    Add-Content -Path $log -Value $line
    Write-Host $line
}

function Test-Healthy {
    # 1. process
    if (-not (Get-Process llama-swap -ErrorAction SilentlyContinue)) {
        return 'llama-swap process not running'
    }

    # 2. API
    try {
        $models = Invoke-RestMethod -Uri "$Api/v1/models" -TimeoutSec 20
    } catch {
        return "API not responding: $($_.Exception.Message)"
    }

    # 3. resident models present and loaded.
    # An on-demand model being up means the resident set was legitimately
    # evicted; treat that as healthy rather than restarting mid-task.
    $onDemandUp = $models.data |
        Where-Object { $_.id -in @('bonsai-vision', 'critic') -and $_.status.value -eq 'loaded' }
    if (-not $onDemandUp) {
        foreach ($m in $Resident) {
            $row = $models.data | Where-Object { $_.id -eq $m }
            if (-not $row) { return "model '$m' missing from /v1/models" }
            if ($row.status.value -ne 'loaded') { return "model '$m' status=$($row.status.value)" }
        }
    }

    # 4. code-intelligence HTTP API. Restarting the stack relaunches it,
    # since start-stack.bat spawns it alongside llama-swap.
    try {
        $h = Invoke-RestMethod -Uri "$ToolsApi/health" -TimeoutSec 20
        if (-not $h.ok) { return 'tools API unhealthy' }
    } catch {
        return "tools API not responding: $($_.Exception.Message)"
    }

    # 5. laya decision engine (own venv, own process)
    try {
        $l = Invoke-RestMethod -Uri "$LayaApi/health" -TimeoutSec 20
        if (-not $l.ok) { return 'laya unhealthy' }
    } catch {
        return "laya not responding: $($_.Exception.Message)"
    }

    # 6. the primary can actually produce a token
    try {
        $body = @{
            model      = 'bonsai-agent'
            messages   = @(@{ role = 'user'; content = 'ping' })
            max_tokens = 2
        } | ConvertTo-Json -Depth 5
        $r = Invoke-RestMethod -Uri "$Api/v1/chat/completions" -Method Post `
            -Body $body -ContentType 'application/json' -TimeoutSec 120
        if (-not $r.choices) { return 'generation returned no choices' }
    } catch {
        return "generation failed: $($_.Exception.Message)"
    }

    return $null
}

$problem = Test-Healthy

if (-not $problem) {
    Write-Log 'OK'
    exit 0
}

Write-Log "UNHEALTHY: $problem"

# Cooldown: a crash-on-load model must not become a restart loop.
if (Test-Path $stamp) {
    $last = Get-Content $stamp -Raw
    $lastTime = [datetime]::MinValue
    if ([datetime]::TryParse($last, [ref]$lastTime)) {
        $age = (Get-Date) - $lastTime
        if ($age.TotalMinutes -lt $RestartCooldownMin) {
            Write-Log ("SUPPRESSED: last restart {0:N1} min ago (cooldown {1} min). Not restarting." -f $age.TotalMinutes, $RestartCooldownMin)
            exit 2
        }
    }
}

if ($WhatIfOnly) {
    Write-Log 'WhatIfOnly: would restart now'
    exit 3
}

Write-Log 'restarting llama-stack'
Set-Content -Path $stamp -Value (Get-Date -Format 'o')

Get-Process llama-swap, llama-server -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Get-CimInstance Win32_Process -Filter "Name='wscript.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -like '*start-stack-hidden*' } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
try { Stop-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue } catch {}
Start-Sleep -Seconds 6
Start-ScheduledTask -TaskName $TaskName
Write-Log 'restart issued'
exit 1
