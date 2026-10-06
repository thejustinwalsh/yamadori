# Samples the Windows counters that say where a Flash-Next prefill spends its time (bench/fn_first_prompt.py starts it).
# One JSON line per sample (~1 s apart): epoch seconds, the memory counters, the physical disks' read rate and each
# llama-server process's IO read rate, working set, private bytes, page faults and PID. Runs until -StopFile exists.
# Counters are the system's own (Get-Counter); nothing is estimated here.
param(
    [Parameter(Mandatory = $true)][string]$Out,
    [Parameter(Mandatory = $true)][string]$StopFile
)
$paths = @(
    '\Memory\Available MBytes',
    '\Memory\Free & Zero Page List Bytes',
    '\Memory\Standby Cache Normal Priority Bytes',
    '\Memory\Standby Cache Reserve Bytes',
    '\Memory\Standby Cache Core Bytes',
    '\Memory\Modified Page List Bytes',
    '\Memory\Page Reads/sec',
    '\Memory\Pages Input/sec',
    '\Memory\Page Faults/sec',
    '\Memory\Transition Faults/sec',
    '\Memory\Cache Faults/sec',
    '\Memory\Demand Zero Faults/sec',
    '\Memory\Committed Bytes',
    '\PhysicalDisk(*)\Disk Read Bytes/sec',
    '\Process(llama-server*)\IO Read Bytes/sec',
    '\Process(llama-server*)\Working Set',
    '\Process(llama-server*)\Private Bytes',
    '\Process(llama-server*)\ID Process',
    '\Process(llama-server*)\Page Faults/sec'
)
$epoch = [datetime]'1970-01-01T00:00:00Z'
$fh = [System.IO.StreamWriter]::new($Out, $false, [System.Text.UTF8Encoding]::new($false))
try {
    while (-not (Test-Path -LiteralPath $StopFile)) {
        $r = Get-Counter -Counter $paths -SampleInterval 1 -MaxSamples 1 -ErrorAction SilentlyContinue
        if (-not $r) { continue }
        # the system clock when the sample returned: Get-Counter's own Timestamp ran exactly 5 h ahead of UTC on this
        # box (2026-10-06, the machine is on UTC-4), so it is kept only as t_counter
        $t = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds() / 1000.0
        $tc = ($r.Timestamp.ToUniversalTime() - $epoch).TotalSeconds
        $m = [ordered]@{ t = [math]::Round($t, 3); t_counter = [math]::Round($tc, 3) }
        $procs = @{}
        foreach ($s in $r.CounterSamples) {
            $p = $s.Path
            $v = [math]::Round($s.CookedValue, 1)
            if ($p -match '\\process\((?<i>[^)]+)\)\\(?<c>.+)$') {
                $i = $Matches['i']
                if (-not $procs.ContainsKey($i)) { $procs[$i] = [ordered]@{} }
                $procs[$i][$Matches['c']] = $v
            } elseif ($p -match '\\physicaldisk\((?<i>[^)]+)\)\\') {
                $m['disk_read_bps ' + $Matches['i']] = $v
            } elseif ($p -match '\\memory\\(?<c>.+)$') {
                $m[$Matches['c']] = $v
            }
        }
        $m['procs'] = $procs
        $fh.WriteLine(($m | ConvertTo-Json -Compress -Depth 5))
        $fh.Flush()
    }
} finally {
    $fh.Close()
}
