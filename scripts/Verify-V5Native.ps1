param([switch]$WithWorld,[switch]$DiagnoseOOM,[switch]$ActivationOffload)
$ErrorActionPreference = 'Stop'
$probeRoot = Split-Path -Parent $PSScriptRoot
$probeLog = Join-Path $probeRoot 'logs/v5-qwen-drive-world'
$tag = if ($WithWorld) { 'native-joint-cap85-updates2' } else { 'native-cap85-updates2' }
$probeArgs = @('-m', 'v5_qwen_drive_world.verify_accumulation', '--allocator-fraction', '0.85')
if ($WithWorld) { $probeArgs += '--with-world' }
if ($DiagnoseOOM) { $probeArgs += '--diagnose-oom'; $tag += '-trace' }
if ($ActivationOffload) { $probeArgs += '--activation-offload'; $tag += '-offload' }
$tag += '-fla-foreach-tiled'
if ($WithWorld) { $tag += '-blocks-convtiles-encoder' }
$probe = Start-Process -FilePath (Join-Path $probeRoot '.venv-train/Scripts/python.exe') -ArgumentList $probeArgs -WorkingDirectory $probeRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $probeLog "accumulation-$tag.stdout.log") -RedirectStandardError (Join-Path $probeLog "accumulation-$tag.stderr.log")
$watch = [Diagnostics.Stopwatch]::StartNew()
while (-not $probe.HasExited) {
    $mem = Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory
    $gpu = @(Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUAdapterMemory | Select-Object Name,DedicatedUsage,SharedUsage)
    $phase = try { (Get-Content -LiteralPath (Join-Path $probeRoot "artifacts/v5-qwen-drive-world/accumulation-$tag.json") -Raw | ConvertFrom-Json).phase } catch { 'starting' }
    [ordered]@{time=(Get-Date).ToString('o'); elapsed_s=[math]::Round($watch.Elapsed.TotalSeconds,2); phase=$phase; available_MB=$mem.AvailableMBytes; committed_bytes=$mem.CommittedBytes; commit_limit=$mem.CommitLimit; gpu=$gpu} | ConvertTo-Json -Depth 4 -Compress | Add-Content -LiteralPath (Join-Path $probeLog "accumulation-$tag-memory.jsonl")
    if ($watch.Elapsed.TotalSeconds -gt 1800) {
        Stop-Process -Id $probe.Id
        throw 'This bounded verification exceeded 1800 seconds; stopped its own process.'
    }
    Start-Sleep -Seconds 2
    $probe.Refresh()
}
$probe.WaitForExit()
$reportPath = Join-Path $probeRoot "artifacts/v5-qwen-drive-world/accumulation-$tag.json"
if (Test-Path -LiteralPath $reportPath) {
    $report = Get-Content -LiteralPath $reportPath -Raw | ConvertFrom-Json
    $report | Add-Member -NotePropertyName process_exit_code -NotePropertyValue $probe.ExitCode -Force
    if ($report.phase -notin @('passed','failed') -and $report.phase -notlike 'terminated*') {
        $report | Add-Member -NotePropertyName last_recorded_phase -NotePropertyValue $report.phase -Force
        $report.phase = 'native_process_exit'
    }
    $report | ConvertTo-Json -Depth 30 | Set-Content -LiteralPath $reportPath -Encoding utf8
}
Write-Output "Verification exit: $($probe.ExitCode)"
exit $probe.ExitCode
