# ONE runner at a time (operator, 2026-09-23). Stages run in order; resume
# redoes stack_error pairs; a STOP file in a run dir stops that runner
# cleanly between rows, and CHAIN-STOP (here) ends the chain after it.
#   A6 (everything forced) on t1b then t1a; then A3 (deep thinking only) on
#   three_tsl, then typegpu,react; then the remaining A0,S0 pairs of t1b and
#   t1a (including the ts03 S0 redo). Order set by the coordinator 10:2x.
# -Concurrent "" once the proxy enforces MAIN_LANES=1 (it serialises every
# request itself, so rows are single-request).
param([string]$Concurrent = "")
$env:PYTHONIOENCODING = 'utf-8'
$py = 'C:\Users\jwals\textgen\installer_files\env\python.exe'
$key = 'C:\Users\jwals\AppData\Local\Temp\claude\C--Users-jwals-llama-stack\d16e1f09-4699-483b-a257-1ba77b329ce7\scratchpad\dogfood.key'
$res = 'C:\Users\jwals\llama-stack\bench\domain\results'
$reason = '"run_tests: full suite passed 41/41 at 02:24; since then only bench/domain run.py, analyse.py, test_run.py changed (test_run 205/205, ruff clean, 17:3x). live_stack: not run while other benchmarks share the single lane"'
Set-Location 'C:\Users\jwals\llama-stack'
$stages = @(
    @('overnight-0923-t1b', 'rust_wasm,three_tsl', 'A6'),
    @('overnight-0923-t1a', 'typescript,typegpu,react', 'A6'),
    @('overnight-0923-t1b', 'three_tsl', 'A3'),
    @('overnight-0923-t1a', 'typegpu,react', 'A3'),
    @('overnight-0923-t1b', 'rust_wasm,three_tsl', 'A0,S0'),
    @('overnight-0923-t1a', 'typescript,typegpu,react', 'A0,S0')
)
foreach ($st in $stages) {
    $rid = $st[0]; $doms = $st[1]; $arms = $st[2]
    $argv = @('bench/domain/run.py', '--key-file', $key, '--suite', 'core,react',
              '--arms', $arms, '--effort', 'medium', '--temperature', '1.0',
              '--sample-per-domain', '8', '--seed', '20260923',
              '--group', 'overnight-0923-t1', '--timeout', '14400',
              '--skip-preflight', 'run_tests,live_stack', '--skip-reason', $reason,
              '--condition-epoch', '2026-09-23 17:24:27',
              '--run-id', $rid, '--domains', $doms)
    if ($Concurrent) { $argv += @('--concurrent-with', $Concurrent) }
    & $py @argv 2>&1 | Out-File -Encoding utf8 -Append (Join-Path $res "$rid.chain.log")
    if (Test-Path "$res\CHAIN-STOP") { Remove-Item "$res\CHAIN-STOP"; break }
}
