<#
.SYNOPSIS
    Register llama-stack to start hidden at logon, and retire the old
    text-generation-webui autostart.

.DESCRIPTION
    Creates the "llama-stack" Scheduled Task. The task runs a VBS shim so no
    console window appears; all output goes to logs\stack.log.

    Needs no elevation: it runs as the current interactive user.
#>
[CmdletBinding()]
param(
    [switch]$Uninstall
)

$ErrorActionPreference = 'Stop'
$root = 'C:\Users\jwals\llama-stack'
$vbs = Join-Path $root 'scripts\start-stack-hidden.vbs'
$me = "$env:USERDOMAIN\$env:USERNAME"

if ($Uninstall) {
    Unregister-ScheduledTask -TaskName 'llama-stack' -Confirm:$false -ErrorAction SilentlyContinue
    Write-Host 'llama-stack task removed'
    return
}

if (-not (Test-Path $vbs)) { throw "launcher missing: $vbs" }

# The old stack must not also grab port 1234 at logon.
$old = Get-ScheduledTask -TaskName 'text-generation-webui' -ErrorAction SilentlyContinue
if ($old) {
    Disable-ScheduledTask -TaskName 'text-generation-webui' | Out-Null
    Write-Host 'text-generation-webui autostart -> DISABLED (task kept, re-enable any time)'
}

$action = New-ScheduledTaskAction -Execute 'wscript.exe' `
    -Argument ('"{0}"' -f $vbs) -WorkingDirectory $root

$trigger = New-ScheduledTaskTrigger -AtLogOn -User $me
$trigger.Delay = 'PT30S'

$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries -StartWhenAvailable `
    -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1) `
    -MultipleInstances IgnoreNew -Hidden

$principal = New-ScheduledTaskPrincipal -UserId $me -LogonType Interactive -RunLevel Limited

Register-ScheduledTask -TaskName 'llama-stack' -Action $action -Trigger $trigger `
    -Settings $settings -Principal $principal -Force `
    -Description 'llama-swap: Bonsai 2 27B ternary x2 + embeddings + reranker. OpenAI API and control UI on :1234.' | Out-Null

Write-Host ''
Get-ScheduledTask -TaskName 'llama-stack', 'text-generation-webui' -ErrorAction SilentlyContinue |
    Select-Object TaskName, State | Format-Table -AutoSize
Write-Host 'Start now with:  Start-ScheduledTask -TaskName llama-stack'

# ---------------------------------------------------------------------------
# Watchdog: health check every 5 minutes, self-heals on failure.
# Separate task so it survives even if the stack task itself is stopped.
# ---------------------------------------------------------------------------
$wdScript = Join-Path $root 'scripts\watchdog.ps1'
$wdAction = New-ScheduledTaskAction -Execute 'powershell.exe' `
    -Argument ('-NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File "{0}"' -f $wdScript) `
    -WorkingDirectory $root

# At logon (after the stack has had time to come up) and every 5 minutes after.
$wdTrigger = New-ScheduledTaskTrigger -AtLogOn -User $me
$wdTrigger.Delay = 'PT3M'
$wdRepeat = New-ScheduledTaskTrigger -Once -At (Get-Date).Date.AddMinutes(1) `
    -RepetitionInterval (New-TimeSpan -Minutes 5)

$wdSettings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 10) `
    -MultipleInstances IgnoreNew -Hidden

Register-ScheduledTask -TaskName 'llama-stack-watchdog' -Action $wdAction `
    -Trigger @($wdTrigger, $wdRepeat) -Settings $wdSettings -Principal $principal -Force `
    -Description 'Health-checks llama-stack every 5 min and restarts it if unhealthy (10 min restart cooldown).' | Out-Null

Write-Host ''
Get-ScheduledTask -TaskName 'llama-stack', 'llama-stack-watchdog', 'text-generation-webui' -ErrorAction SilentlyContinue |
    Select-Object TaskName, State | Format-Table -AutoSize
