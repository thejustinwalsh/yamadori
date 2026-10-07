# The 2026-10-07 package skills probe (coordinator: "RUN THE PROBE ... detached"): waits for
# pid 18908 (the skills agent's overnight chain) to exit, then runs all four arms
# (control / inject / router / both) x 5 trials x 15 steps, then --report. The arms force
# their own header (the control sends package_skills false: the channel is ON by default
# since 9398750). Key: .keys\live-test.key (never printed). A preflight refusal (exit 3:
# GPU busy) is retried 5 times, 120 s apart, then the script stops. Detached (Start-Process);
# logs to $Log.
param([int]$WaitPid = 18908)
$ErrorActionPreference = "Continue"
$Root = "C:\Users\jwals\llama-stack"
$Py = "C:\Users\jwals\textgen\installer_files\env\python.exe"
$Out = "$Root\bench\mcp\results"
$Log = "$Out\package_skills_probe_run.log"
$Key = "$Root\.keys\live-test.key"
Set-Location $Root
New-Item -ItemType Directory -Force $Out | Out-Null
function Say($m) { "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $m | Out-File -FilePath $Log -Append -Encoding utf8 }

Say "script started; waiting for pid $WaitPid"
while (Get-Process -Id $WaitPid -ErrorAction SilentlyContinue) { Start-Sleep -Seconds 30 }
Say "pid $WaitPid has exited"
for ($i = 1; $i -le 6; $i++) {
    Say "RUN attempt $i"
    & $Py "bench/mcp/package_skills_probe.py" --run --trials 5 --steps 15 --max-minutes 20 --tag pkgskills --key-file $Key 2>&1 | Out-File -FilePath $Log -Append -Encoding utf8
    $code = $LASTEXITCODE
    Say "RUN exit=$code"
    if ($code -ne 3) { break }
    Say "preflight refused; retrying in 120 s"
    Start-Sleep -Seconds 120
}
& $Py "bench/mcp/package_skills_probe.py" --report 2>&1 | Out-File -FilePath "$Out\package_skills_probe_report.json" -Encoding utf8
Say "report written; done"
