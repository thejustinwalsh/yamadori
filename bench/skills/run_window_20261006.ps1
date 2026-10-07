# The 2026-10-06 injector window (coordinator: "GPU go, gated"): waits for the
# Decision Index run (pid 98872) to exit, then reads variants f,g,h and the 8
# B-only cases of a-e on bonsai-a4000, then the offline analyses. Sends nothing
# to the main card. Detached: started with Start-Process, logs to $Log.
param([int]$WaitPid = 98872)
$ErrorActionPreference = "Continue"
$Root = "C:\Users\jwals\llama-stack"
$Py = "C:\Users\jwals\textgen\installer_files\env\python.exe"
$Cases = "C:/Users/jwals/octo/codex-label-kit/cases.jsonl"
$Out = "$Root\bench\skills\inject\results"
$Log = "$Out\window_20261006.log"
$DiStatus = "C:\Users\jwals\octo\decision-index\runs\jjava-strat860\status.json"
Set-Location $Root

function Say($m) { "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $m | Out-File -FilePath $Log -Append -Encoding utf8 }
function Step($name, $argList, $outFile) {
    Say "START $name"
    if ($outFile) {
        & $Py @argList 1> $outFile 2>> $Log
    } else {
        & $Py @argList 2>&1 | Out-File -FilePath $Log -Append -Encoding utf8
    }
    $code = $LASTEXITCODE
    Say "END $name exit=$code"
    return $code
}

Say "window script started; waiting for pid $WaitPid"
while (Get-Process -Id $WaitPid -ErrorAction SilentlyContinue) { Start-Sleep -Seconds 30 }
Say "pid $WaitPid has exited"
if (Test-Path $DiStatus) { Say ("decision-index status.json: " + (Get-Content $DiStatus -Raw)) } else { Say "decision-index status.json absent" }

# inject_decide refuses (exit 2) unless bonsai-a4000 is loaded and idle; it never
# loads a model. Retry for up to an hour, then go on and say so.
function Decide($variants) {
    for ($i = 0; $i -lt 60; $i++) {
        $code = Step "inject_decide $variants" @("bench/skills/inject_decide.py", "--cases", $Cases, "--model", "bonsai-a4000", "--variants", $variants, "--only-labelled", "--skip-done") $null
        if ($code -ne 2) { return $code }
        Say "refused (model not loaded or card busy); retrying in 60 s ($($i + 1)/60)"
        Start-Sleep -Seconds 60
    }
    Say "GAVE UP on inject_decide $variants after 60 refusals"
    return 2
}
$null = Decide "f,g,h"
$null = Decide "a,b,c,d,e"
$null = Step "inject_diagnose" @("bench/skills/inject_diagnose.py", "--cases", $Cases, "--model", "bonsai-a4000") "$Out\diagnose_20261006.json"
$null = Step "inject_tune plain" @("bench/skills/inject_tune.py", "--model", "bonsai-a4000", "--json") "$Out\tune_plain_20261006.json"
$null = Step "inject_tune loro (all variants)" @("bench/skills/inject_tune.py", "--model", "bonsai-a4000", "--loro") "$Out\tune_loro_all_20261006.json"
foreach ($v in "a", "d", "f", "g", "h") {
    $null = Step "inject_tune loro $v" @("bench/skills/inject_tune.py", "--model", "bonsai-a4000", "--loro", "--variant", $v) "$Out\tune_loro_${v}_20261006.json"
}
Say "WINDOW DONE"
