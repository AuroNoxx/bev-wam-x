param([int]$TargetId, [string]$OutputFile)
$ErrorActionPreference = 'Stop'
$query = 'Win32_PerfFormattedData_GPUPerformanceCounters_GPUProcessMemory'
while (Get-Process -Id $TargetId -ErrorAction SilentlyContinue) {
    $counters = @(Get-CimInstance $query -Filter "Name LIKE 'pid`_$TargetId`_%'" |
        Where-Object { $_.Name.StartsWith("pid_${TargetId}_") })
    $entry = @{
        utc = [datetime]::UtcNow.ToString('o')
        pid = $TargetId
        instances = $counters.Count
        dedicated_bytes = ($counters | Measure-Object -Property DedicatedUsage -Sum).Sum
        shared_bytes = ($counters | Measure-Object -Property SharedUsage -Sum).Sum
    }
    ($entry | ConvertTo-Json -Compress) | Add-Content -LiteralPath $OutputFile
    Start-Sleep -Seconds 1
}
