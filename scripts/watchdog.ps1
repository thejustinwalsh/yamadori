<#
.SYNOPSIS
    Liveness check and self-heal for the llama-stack.

.DESCRIPTION
    Three states, and nothing else:

      running   every /health route answers   -> do nothing
      down      a /health route does not      -> restart that service
      missing   the process is not there      -> restart that service

    It does NOT inspect model names, model status, or generate a token. That
    is the whole point of this rewrite.

    WHAT THE OLD VERSION DID, AND WHAT IT COST

    It asked the PROXY on :1234 for a model named 'bonsai'. The proxy
    deliberately advertises one model, 'yamadori', because internal names are
    implementation detail. So every pass found 'bonsai' missing, concluded the
    stack was broken, and killed llama-swap and every llama-server with it:

        23:51:02  UNHEALTHY: model 'bonsai' missing  -> restarting
        00:06:02  UNHEALTHY: model 'bonsai' missing  -> restarting
        00:16:02  UNHEALTHY: model 'bonsai' missing  -> restarting

    Every ten minutes, for hours, against a completely healthy stack. It cost
    a LiveCodeBench run -- roughly 60% of rows recorded as generation errors,
    spread evenly through the file because the kills were on a timer -- and
    hours spent attributing those to the model, then the transport, then
    llama-swap, each of which was innocent.

    It also ran a generation probe with a 120-second timeout. One answer on
    this model legitimately takes 450 seconds, so a busy stack failed its own
    health check and was killed for being busy.

    Both failures have the same root: the watchdog knew things about the system
    that the system was free to change. A liveness check that understands the
    application will eventually be wrong about the application, and being wrong
    here means killing it. So this one understands nothing. It asks each
    service whether it is answering, and that is all it is allowed to know.

    TWO GUARDS AGAINST KILLING A BUSY STACK (2026-09-22)

      two strikes   nothing is restarted unless it was ALSO unhealthy on the
                    previous run, 5 minutes earlier. One slow probe during a
                    busy moment can never kill anything.
      progress      before restarting llama-swap -- which kills every
                    generation on every slot -- the llama-server slot counters
                    are read twice, 10 s apart. Moving counters mean tokens are
                    being produced: alive, not restarted. Measured live under a
                    20k-token thinking run plus worker extraction.

    Neither needs to understand the application: "a probe failed twice" and
    "tokens are coming out" are both plain liveness facts.

    A service that answers /health while returning wrong answers is a bug for a
    test to catch, not for a watchdog to restart. Restarting cannot fix a wrong
    answer, and trying is how a monitor becomes the outage.
#>
[CmdletBinding()]
param(
    [string]$TaskName = 'llama-stack',
    [int]$RestartCooldownMin = 10,
    [int]$TimeoutSec = 15,
    [switch]$WhatIfOnly
)

$ErrorActionPreference = 'Continue'
$root = Split-Path -Parent $PSScriptRoot
$log = Join-Path $root 'logs\watchdog.log'
$stamp = Join-Path $root 'logs\.last-restart'
$py = 'C:\Users\jwals\textgen\installer_files\env\python.exe'
New-Item -ItemType Directory -Force -Path (Split-Path $log) | Out-Null

# The services, and how each one comes back.
#
# llama-swap owns the GPU and every llama-server under it, and it is started
# by the Scheduled Task, so bringing it back means restarting the task. The
# other three are single commands and are restarted on their own -- reloading
# 6 GB of weights because the decision engine died would be a worse outage
# than the one being repaired.
#
# Launch lines are duplicated from scripts/start-stack.bat, which is a real
# risk: change one and this silently starts something different. They are kept
# here anyway because the alternative is a full stack restart for a 400 MB
# Python process. If start-stack.bat changes, change these.
$Services = @(
    @{ Name = 'llama-swap'; Url = 'http://127.0.0.1:11434/health'; Kind = 'task'
       Process = 'llama-swap' }
    @{ Name = 'proxy';      Url = 'http://127.0.0.1:1234/health';  Kind = 'process'
       Exe = $py; Args = @("$root\mcp\server.py"); Log = "$root\logs\proxy.log"
       Match = 'mcp[\\/]server\.py' }
    @{ Name = 'tools-api';  Url = 'http://127.0.0.1:1235/health';  Kind = 'process'
       Exe = $py; Args = @("$root\mcp\tools_api.py"); Log = "$root\logs\tools-api.log"
       Match = 'tools_api\.py' }
    @{ Name = 'laya';       Url = 'http://127.0.0.1:1237/health';  Kind = 'process'
       Exe = "$root\.venv-laya\Scripts\python.exe"; Args = @("$root\mcp\laya_service.py")
       Log = "$root\logs\laya.log"; Match = 'laya_service\.py' }
    # The job worker has no port: it claims rows from index/jobs.sqlite3. Alive
    # means its process exists. A worker that is alive but stuck is not this
    # script's business -- jobs.reclaim() hands a stale job to the next worker.
    @{ Name = 'worker';     Url = $null;                             Kind = 'process'
       Exe = $py; Args = @("$root\mcp\worker.py"); Log = "$root\logs\worker.log"
       Match = 'mcp[\\/]worker\.py' }
)

function Write-Log([string]$msg) {
    $line = "{0}  {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $msg
    Add-Content -Path $log -Value $line
    Write-Host $line
}

function Test-Alive($svc) {
    # A service with no URL (the job worker) is alive if its process exists.
    if (-not $svc.Url) {
        return [bool](Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
            Where-Object { $_.CommandLine -match $svc.Match })
    }
    # Answering is the only question. Any 2xx counts; the body is not read,
    # because reading it is how a watchdog starts having opinions.
    try {
        $r = Invoke-WebRequest -Uri $svc.Url -UseBasicParsing -TimeoutSec $TimeoutSec
        return ($r.StatusCode -ge 200 -and $r.StatusCode -lt 300)
    } catch {
        return $false
    }
}

function Test-UpstreamsAlive {
    # llama-swap answering does not mean the MODEL is answering. llama-swap is
    # a Go reverse proxy; a wedged llama-server behind it leaves :11434/health
    # returning 200 forever while nothing can generate.
    #
    # It proxies a health route per upstream, so this stays a /health check.
    # The names come from llama-swap's own /running, never from this file --
    # a hardcoded 'bonsai' here is exactly the coupling that made the previous
    # version kill a healthy stack every ten minutes.
    #
    # Only a TIMEOUT counts as down. A non-2xx is llama-swap answering about a
    # model, which is a statement this script is not qualified to interpret,
    # and interpreting it is how the last rewrite started.
    try {
        $running = Invoke-RestMethod -Uri 'http://127.0.0.1:11434/running' -TimeoutSec $TimeoutSec
    } catch {
        return $null   # llama-swap itself is down; the main check owns that.
    }
    foreach ($m in @($running.running)) {
        $id = $m.model
        if (-not $id) { continue }
        $t0 = Get-Date
        try {
            Invoke-WebRequest -Uri "http://127.0.0.1:11434/upstream/$id/health" `
                -UseBasicParsing -TimeoutSec $TimeoutSec | Out-Null
        } catch {
            if (((Get-Date) - $t0).TotalSeconds -ge ($TimeoutSec - 1)) {
                return "upstream '$id' did not answer its health route in ${TimeoutSec}s"
            }
            Write-Log "  upstream '$id' health route answered non-2xx; not acting on it"
        }
    }
    return $null
}

function Get-SlotProgress {
    # Total tokens decoded + prompt tokens processed across processing slots of
    # every running model, or $null if nothing can be read. This is liveness,
    # not application knowledge: a server whose counters move is producing
    # tokens, whatever its health route says.
    try {
        $running = Invoke-RestMethod -Uri 'http://127.0.0.1:11434/running' -TimeoutSec 5
    } catch { return $null }
    $total = 0; $busy = 0; $read = $false
    foreach ($m in @($running.running)) {
        if (-not $m.model) { continue }
        try {
            $slots = Invoke-RestMethod -Uri "http://127.0.0.1:11434/upstream/$($m.model)/slots" -TimeoutSec 10
            $read = $true
        } catch { continue }
        foreach ($s in @($slots)) {
            if (-not $s.is_processing) { continue }
            $busy++
            $nt = @($s.next_token)
            $dec = if ($nt.Count -gt 0 -and $nt[0].n_decoded) { [int64]$nt[0].n_decoded } else { 0 }
            $total += $dec + [int64]($s.n_prompt_tokens_processed)
        }
    }
    if (-not $read) { return $null }
    return @{ total = $total; busy = $busy }
}

function Test-MakingProgress {
    # Two samples 10 s apart. Any processing slot whose counters moved means
    # the model is working. A long generation must never be killed for being
    # long: the previous watchdog did exactly that with a 120 s probe against
    # answers that take 450 s.
    $a = Get-SlotProgress
    if ($null -eq $a -or $a.busy -eq 0) { return $false }
    Start-Sleep -Seconds 10
    $b = Get-SlotProgress
    if ($null -eq $b) { return $false }
    return ($b.total -ne $a.total)
}

# TWO STRIKES. A service is restarted only if it was also found unhealthy on
# the PREVIOUS run (the task repeats every 5 minutes). One slow probe during a
# busy moment can never kill anything; a service that is really dead waits one
# more interval. The mark expires if the previous run is too old to count.
$suspect = Join-Path $root 'logs\.suspect'
function Test-SecondStrike([string]$name) {
    $f = "$suspect.$name"
    $last = [datetime]::MinValue
    $second = (Test-Path $f) -and [datetime]::TryParse((Get-Content $f -Raw), [ref]$last) -and `
              (((Get-Date) - $last).TotalMinutes -lt 12)
    Set-Content -Path $f -Value (Get-Date -Format 'o')
    return $second
}
function Clear-Strike([string]$name) {
    Remove-Item "$suspect.$name" -ErrorAction SilentlyContinue
}

function Test-CooledDown([string]$name) {
    # Per service, so a thrashing laya cannot suppress a llama-swap restart.
    $f = "$stamp.$name"
    if (-not (Test-Path $f)) { return $true }
    $last = [datetime]::MinValue
    if (-not [datetime]::TryParse((Get-Content $f -Raw), [ref]$last)) { return $true }
    $age = (Get-Date) - $last
    if ($age.TotalMinutes -lt $RestartCooldownMin) {
        Write-Log ("SUPPRESSED {0}: last restart {1:N1} min ago (cooldown {2} min)" -f $name, $age.TotalMinutes, $RestartCooldownMin)
        return $false
    }
    return $true
}

function Restart-Service($svc) {
    Set-Content -Path "$stamp.$($svc.Name)" -Value (Get-Date -Format 'o')
    if ($svc.Kind -eq 'task') {
        Write-Log "restarting the whole stack (Scheduled Task '$TaskName')"
        # Also stamp the legacy single-file marker, so an older copy of this
        # script running from a stale Scheduled Task still honours a cooldown.
        Set-Content -Path $stamp -Value (Get-Date -Format 'o')
        Get-Process llama-swap, llama-server -ErrorAction SilentlyContinue |
            Stop-Process -Force -ErrorAction SilentlyContinue
        Get-CimInstance Win32_Process -Filter "Name='wscript.exe'" -ErrorAction SilentlyContinue |
            Where-Object { $_.CommandLine -like '*start-stack-hidden*' } |
            ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
        try { Stop-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue } catch {}
        Start-Sleep -Seconds 6
        Start-ScheduledTask -TaskName $TaskName
        Write-Log 'stack restart issued'
        return
    }

    # A single service. Kill whatever is left of it first: a process that is
    # alive but not answering still holds the port, and the replacement would
    # fail to bind and leave nothing running at all.
    if ($svc.Match) {
        Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
            Where-Object { $_.CommandLine -match $svc.Match } |
            ForEach-Object {
                Write-Log "  stopping stale $($svc.Name) pid $($_.ProcessId)"
                Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
            }
        Start-Sleep -Seconds 2
    }
    if (-not (Test-Path $svc.Exe)) {
        Write-Log "  cannot restart $($svc.Name): interpreter missing at $($svc.Exe)"
        return
    }
    Write-Log "restarting $($svc.Name)"
    Start-Process -FilePath $svc.Exe -ArgumentList $svc.Args `
        -WorkingDirectory $root -WindowStyle Hidden `
        -RedirectStandardOutput $svc.Log -RedirectStandardError "$($svc.Log).err"
    Write-Log "  $($svc.Name) restart issued"
}

$down = @()
foreach ($svc in $Services) {
    if ($svc.Process -and -not (Get-Process $svc.Process -ErrorAction SilentlyContinue)) {
        Write-Log "MISSING: $($svc.Name) process not running"
        $down += $svc
        continue
    }
    if (-not (Test-Alive $svc)) {
        Write-Log "DOWN: $($svc.Name) did not answer $($svc.Url)"
        $down += $svc
        continue
    }
    # llama-swap answering is not the same as the models behind it answering.
    if ($svc.Kind -eq 'task') {
        $wedged = Test-UpstreamsAlive
        if ($wedged) {
            Write-Log "WEDGED: $wedged"
            $down += $svc
        }
    }
}

foreach ($svc in $Services) {
    if (-not ($down | Where-Object { $_.Name -eq $svc.Name })) { Clear-Strike $svc.Name }
}

# A model server whose tokens are moving is alive, whatever else timed out.
# Checked before anything that would restart llama-swap, because restarting it
# kills every generation in flight on every slot.
if ($down | Where-Object { $_.Kind -eq 'task' }) {
    if (Test-MakingProgress) {
        Write-Log "  llama-swap check failed but slot counters are moving -- generating, not dead; NOT restarting"
        $down = @($down | Where-Object { $_.Kind -ne 'task' })
        Clear-Strike 'llama-swap'
    }
}

# First strike: record it and act on nothing.
$confirmed = @()
foreach ($svc in $down) {
    if (Test-SecondStrike $svc.Name) {
        $confirmed += $svc
    } else {
        Write-Log "  $($svc.Name): first strike -- will restart only if still unhealthy on the next run"
    }
}
$down = $confirmed

if ($down.Count -eq 0) {
    Write-Log 'OK'
    exit 0
}

# llama-swap restarts the whole task, which brings the others with it. Doing
# both would kill the replacements a second later.
if ($down | Where-Object { $_.Kind -eq 'task' }) {
    $down = @($down | Where-Object { $_.Kind -eq 'task' })
}

if ($WhatIfOnly) {
    Write-Log ("WhatIfOnly: would restart {0}" -f (($down | ForEach-Object { $_.Name }) -join ', '))
    exit 3
}

$acted = 0
foreach ($svc in $down) {
    if (Test-CooledDown $svc.Name) { Restart-Service $svc; $acted++ }
}
exit $(if ($acted) { 1 } else { 2 })
