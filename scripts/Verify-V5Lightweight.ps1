param([ValidateSet(1,2,4)][int]$Microbatch=4)
$ErrorActionPreference='Stop'
$root=Split-Path -Parent $PSScriptRoot
$tag="lightweight-mb$Microbatch"
$proc=Start-Process -FilePath "$root/.venv-train/Scripts/python.exe" -ArgumentList @('-m','v5_qwen_drive_world.verify_lightweight','--microbatch',"$Microbatch") -WorkingDirectory $root -WindowStyle Hidden -PassThru -RedirectStandardOutput "$root/logs/v5-qwen-drive-world/$tag.stdout.log" -RedirectStandardError "$root/logs/v5-qwen-drive-world/$tag.stderr.log"
$watch=[Diagnostics.Stopwatch]::StartNew()
while(-not $proc.HasExited) {
    $mem=Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory
    $gpu=@(Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUAdapterMemory | Select-Object Name,DedicatedUsage,SharedUsage)
    @{elapsed_s=$watch.Elapsed.TotalSeconds;available_MB=$mem.AvailableMBytes;gpu=$gpu} | ConvertTo-Json -Depth 4 -Compress | Add-Content "$root/logs/v5-qwen-drive-world/$tag-memory.jsonl"
    if($watch.Elapsed.TotalSeconds -gt 900) {
        Get-CimInstance Win32_Process | Where-Object ParentProcessId -eq $proc.Id | ForEach-Object {Stop-Process -Id $_.ProcessId}
        Stop-Process -Id $proc.Id
        throw 'Bounded verification timed out after900s'
    }
    Start-Sleep -Seconds 2;$proc.Refresh()
}
$proc.WaitForExit()
Write-Output "Verification exit: $($proc.ExitCode)"
exit $proc.ExitCode
