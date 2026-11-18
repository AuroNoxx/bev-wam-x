# User-run cleanup for the v5 migration. Default: preview. Add -Apply to delete.
[CmdletBinding()]
param([switch]$Apply)

$ErrorActionPreference = 'Stop'
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..')).TrimEnd('\')
if (-not (Test-Path -LiteralPath (Join-Path $projectRoot 'AGENTS.md')) -or
    -not (Test-Path -LiteralPath (Join-Path $projectRoot 'v5_qwen_drive_world'))) {
    throw 'Project identity check failed. Keep this script in highway-occ-flow/scripts.'
}

# Explicit obsolete paths only. Never delete data/public, simulator installations,
# .venv-train, runtime/python*, the official source, v5 assets, logs or reports.
$obsoletePaths = [ordered]@{
    '.cache/carla-0915' = 'Installed CARLA archive and duplicate download chunks'
    '.cache/carla-0915-maps' = 'Old map download chunks'
    '.cache/downloads' = 'Old incomplete installer downloads'
    '.cache/uv' = 'Re-downloadable old package cache'
    '.cache/inductor' = 'Rebuildable old compiled graphs'
    '.cache/triton' = 'Rebuildable old kernel cache'
    '.cache/lightning-bench' = 'Old framework benchmark cache'
    '.cache/lightning-template' = 'Old framework template cache'
    '.cache/modelscope' = 'Old model download cache'
    '.cache/temp' = 'Old project temporary files'
    '.venv-metadrive043' = 'MetaDrive environment unused by the CARLA v5 pipeline'
    'artifacts/qwen-driving-v3/cache' = 'Obsolete v3 frozen model features'
    'artifacts/qwen-driving-v4/cache' = 'Obsolete v4 frozen model features'
    'data/native_occ_aligned_cache' = 'Rebuildable cache for retired native OCC models'
    'data/native_occ_aligned_cache_pilot' = 'Retired OCC cache pilot'
    'data/carla_matched6_cache_v1' = 'Old derived training cache'
    'data/carla_recovery_cache_v1' = 'Old derived training cache'
    'data/carla_reactive_hold_cache_v1' = 'Old derived training cache'
    'data/carla_startstop_cache_v1' = 'Old derived training cache'
    'data/carla_recovery_dev_cache_v2' = 'Old derived training cache'
    'data/carla_miniloop_cache_v1' = 'Old derived training cache'
    'data/carla_recovery_combined_v2' = 'Old derived cache index'
}

function Assert-ProjectPath([string]$Candidate) {
    $absolute = [IO.Path]::GetFullPath($Candidate)
    if (-not $absolute.StartsWith($projectRoot + '\', [StringComparison]::OrdinalIgnoreCase)) {
        throw "Path outside project: $absolute"
    }
    # Check every existing ancestor; a junction in a parent can escape the project.
    $cursor = $absolute
    while ($cursor.Length -ge $projectRoot.Length) {
        if (Test-Path -LiteralPath $cursor) {
            $item = Get-Item -LiteralPath $cursor -Force
            if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) {
                throw "Link/junction is not eligible for recursive cleanup: $cursor"
            }
        }
        if ($cursor -eq $projectRoot) { break }
        $cursor = [IO.Path]::GetDirectoryName($cursor)
    }
    return $absolute
}

function Assert-NoDependentProcess {
    $active = @(Get-CimInstance Win32_Process | Where-Object {
        $_.ProcessId -ne $PID -and
        $_.Name -match '^(python|pythonw|Carla|CarlaUE4|CarlaUnreal).*\.exe$' -and
        $_.CommandLine -and
        $_.CommandLine.IndexOf($projectRoot, [StringComparison]::OrdinalIgnoreCase) -ge 0 -and
        $_.CommandLine -notmatch 'tensorboard\.main|v5_qwen_drive_world\.download'
    })
    if ($active.Count) {
        $active | Select-Object ProcessId, Name, CommandLine | Format-List | Out-Host
        throw 'A project Python/simulator task may use these files. No process was stopped; retry after it finishes.'
    }
}

Assert-NoDependentProcess
$inventory = @(
    foreach ($entry in $obsoletePaths.GetEnumerator()) {
        $absolute = Assert-ProjectPath (Join-Path $projectRoot $entry.Key)
        if (-not (Test-Path -LiteralPath $absolute)) { continue }
        $items = @(Get-ChildItem -LiteralPath $absolute -Recurse -Force)
        if (@($items | Where-Object { $_.Attributes -band [IO.FileAttributes]::ReparsePoint }).Count) {
            throw "Nested link/junction found; no deletion performed: $absolute"
        }
        $bytes = ($items | Where-Object { -not $_.PSIsContainer } | Measure-Object Length -Sum).Sum
        [pscustomobject]@{
            relative_path = $entry.Key
            path = $absolute
            bytes = [long]$bytes
            GB = [math]::Round($bytes / 1e9, 3)
            reason = $entry.Value
            status = 'planned'
        }
    }
)
$inventory | Select-Object relative_path, GB, reason | Format-Table -AutoSize | Out-Host
$totalBytes = ($inventory | Measure-Object bytes -Sum).Sum
Write-Host ('Total logical file size: {0:N2} GB across {1} directories.' -f ($totalBytes / 1e9), $inventory.Count)
Write-Host 'Shared hardlinks can make actual freed disk space smaller than logical file size.'
if (-not $Apply) {
    Write-Host 'Preview only. To delete, rerun this script with -Apply.'
    return
}

Assert-NoDependentProcess
$recordDir = Join-Path $projectRoot 'artifacts/v5-qwen-drive-world'
New-Item -ItemType Directory -Path $recordDir -Force | Out-Null
$recordPath = Join-Path $recordDir ('user-cleanup-{0}.json' -f (Get-Date -Format 'yyyyMMdd-HHmmss'))
$driveInfo = [IO.DriveInfo]::new([IO.Path]::GetPathRoot($projectRoot))
$freeBefore = $driveInfo.AvailableFreeSpace
$report = [ordered]@{
    started = (Get-Date -Format o)
    project = $projectRoot
    logical_bytes = $totalBytes
    free_before = $freeBefore
    entries = $inventory
}
function Write-Record {
    $report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $recordPath -Encoding UTF8
}
Write-Record
foreach ($entry in $inventory) {
    try {
        $verifiedPath = Assert-ProjectPath $entry.path
        # One native PowerShell operation; no alternate deletion methods on failure.
        Remove-Item -LiteralPath $verifiedPath -Recurse -Force -ErrorAction Stop
        if (Test-Path -LiteralPath $verifiedPath) { throw "Path still exists: $verifiedPath" }
        $entry.status = 'deleted'
        Write-Host ('Deleted: ' + $entry.relative_path)
        Write-Record
    } catch {
        $entry.status = 'failed: ' + $_.Exception.Message
        $report['stopped'] = Get-Date -Format o
        Write-Record
        Write-Host ('Stopped. Details: ' + $recordPath)
        throw
    }
}
$report['completed'] = Get-Date -Format o
$report['free_after'] = ([IO.DriveInfo]::new([IO.Path]::GetPathRoot($projectRoot))).AvailableFreeSpace
$report['disk_free_change_bytes'] = $report.free_after - $freeBefore
Write-Record
Write-Host ('Finished. Observed disk free-space change: {0:N2} GB.' -f ($report.disk_free_change_bytes / 1e9))
Write-Host 'Concurrent v5 downloads can reduce the observed free-space change.'
Write-Host ('Record: ' + $recordPath)
