# User-authorized legacy cleanup, 2026-11-15. Run manually; -Preview never deletes.
[CmdletBinding()]
param([switch]$Preview)

$ErrorActionPreference = 'Stop'
$root = 'E:\ai\projects\highway-occ-flow'
if ((Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path -ne $root) {
    throw 'This script must remain in highway-occ-flow\scripts.'
}

# Explicit file allowlist. Never delete an artifacts parent directory.
$oldFiles = @(
    'artifacts\benchmark\flow-b4\checkpoint.pt'
    'artifacts\benchmark\flow-b8\checkpoint.pt'
    'artifacts\benchmark\world-b4\checkpoint.pt'
    'artifacts\benchmark\world-b8\checkpoint.pt'
    'artifacts\diagnostics\alignment-v3\legacy-loader-regression\atomic.ckpt'
    'artifacts\diagnostics\alignment-v3\legacy-loader-regression\split.ckpt'
    'artifacts\diagnostics\flow-v3-finish60-regression\atomic.ckpt'
    'artifacts\diagnostics\flow-v3-finish60-regression\split.ckpt'
    'artifacts\lightning\contract-test\atomic.ckpt'
    'artifacts\lightning\contract-test\split.ckpt'
    'artifacts\lightning\flow\best.ckpt'
    'artifacts\lightning\flow\latest.ckpt'
    'artifacts\lightning\flow-v2\best.ckpt'
    'artifacts\lightning\flow-v2\latest.ckpt'
    'artifacts\lightning\stage1-v2-contract\atomic.ckpt'
    'artifacts\lightning\stage1-v2-contract\initial.ckpt'
    'artifacts\lightning\stage1-v2-contract\split.ckpt'
    'artifacts\v1\contract-test\gpu-latest.pt'
    'artifacts\v1\contract-test\latest.pt'
    'artifacts\v1\v1-flow\best.pt'
    'artifacts\v1\v1-flow\latest.pt'
)
$oldDirectories = @('runtime\CARLA-0.9.15', '.venv')
$protectedDirectories = @(
    'artifacts\lightning\flow-v3', '.venv-train', '.venv-carla-ue5',
    'runtime\CARLA-1.0', 'runtime\python-train', 'data', 'logs', 'docs', 'configs'
)
$protectedFiles = @(
    'artifacts\lightning\flow-v3\best.ckpt',
    'artifacts\lightning\flow-v3\latest.ckpt',
    '.venv-train\pyvenv.cfg', '.venv-carla-ue5\pyvenv.cfg'
)

function Assert-RegularPath([string]$Path) {
    $item = Get-Item -LiteralPath $Path -Force
    while ($null -ne $item) {
        if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) {
            throw "Refusing a link or junction: $($item.FullName)"
        }
        if ($item -is [IO.DirectoryInfo]) { $item = $item.Parent }
        else { $item = $item.Directory }
    }
}

function Get-CheckedTreeBytes([string]$Path) {
    $pending = [Collections.Generic.Stack[string]]::new()
    $pending.Push($Path)
    [long]$bytes = 0
    while ($pending.Count -gt 0) {
        foreach ($item in Get-ChildItem -LiteralPath $pending.Pop() -Force) {
            if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) {
                throw "Refusing a tree containing a link: $($item.FullName)"
            }
            if ($item.PSIsContainer) { $pending.Push($item.FullName) }
            else { $bytes += $item.Length }
        }
    }
    return $bytes
}

Assert-RegularPath $root
$hashesBefore = @{}
foreach ($relative in $protectedDirectories) {
    if (-not (Test-Path -LiteralPath (Join-Path $root $relative) -PathType Container)) {
        throw "Required protected directory missing: $relative"
    }
}
foreach ($relative in $protectedFiles) {
    $hashesBefore[$relative] = (Get-FileHash -LiteralPath (Join-Path $root $relative) -Algorithm SHA256).Hash
}

$targets = @(foreach ($relative in @($oldFiles) + @($oldDirectories)) {
    $path = Join-Path $root $relative
    if (-not (Test-Path -LiteralPath $path)) { continue }
    $resolved = (Resolve-Path -LiteralPath $path).Path
    if ($resolved -ne $path -or -not $resolved.StartsWith($root + '\', [StringComparison]::OrdinalIgnoreCase)) {
        throw "Unexpected target path: $resolved"
    }
    foreach ($protected in $protectedDirectories) {
        $keep = Join-Path $root $protected
        if ($resolved -eq $keep -or $resolved.StartsWith($keep + '\', [StringComparison]::OrdinalIgnoreCase)) {
            throw "Target intersects a protected directory: $resolved"
        }
    }
    Assert-RegularPath $resolved
    $item = Get-Item -LiteralPath $resolved -Force
    $isDirectory = $oldDirectories -contains $relative
    if ($item.PSIsContainer -ne $isDirectory) { throw "Unexpected target type: $relative" }
    $bytes = if ($isDirectory) { Get-CheckedTreeBytes $resolved } else { $item.Length }
    [pscustomobject]@{ RelativePath=$relative; FullPath=$resolved; IsDirectory=$isDirectory; Bytes=$bytes }
})

# Never stop processes automatically. TensorBoard in .venv-train is unaffected.
$active = @(Get-CimInstance Win32_Process | Where-Object {
    $process = $_
    $matched = $false
    foreach ($relative in $oldDirectories) {
        $prefix = (Join-Path $root $relative) + '\'
        if (($process.ExecutablePath -and $process.ExecutablePath.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) -or
            ($process.CommandLine -and $process.CommandLine.IndexOf($prefix, [StringComparison]::OrdinalIgnoreCase) -ge 0)) {
            $matched = $true
        }
    }
    $matched
})
if ($active.Count) {
    $active | Select-Object ProcessId, Name, ExecutablePath | Format-Table
    throw 'A legacy process is active. Close it normally, then rerun this script.'
}

$targets | Select-Object RelativePath, @{n='MiB';e={[math]::Round($_.Bytes/1MB,2)}} | Format-Table -AutoSize
$total = ($targets | Measure-Object Bytes -Sum).Sum
Write-Host ('Allowlisted targets: {0}; logical size: {1:N2} GiB' -f $targets.Count, ($total/1GB))
if ($Preview) {
    Write-Host 'PREVIEW ONLY. No files deleted; no cleanup report written.'
    return
}

$reportDir = Join-Path $root ('artifacts\diagnostics\legacy-cleanup-' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff'))
New-Item -ItemType Directory -Path $reportDir | Out-Null
$targets | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $reportDir 'targets.json') -Encoding UTF8
$hashesBefore | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $reportDir 'protected-before.json') -Encoding UTF8
$freeBefore = (Get-PSDrive E).Free
$removed = [Collections.Generic.List[string]]::new()
$failure = $null
try {
    foreach ($target in $targets) {
        # Recheck immediately before each deletion; remove only these exact paths.
        Assert-RegularPath $target.FullPath
        if ($target.IsDirectory) {
            $null = Get-CheckedTreeBytes $target.FullPath
            Remove-Item -LiteralPath $target.FullPath -Recurse -Force
        } else {
            Remove-Item -LiteralPath $target.FullPath -Force
        }
        $removed.Add($target.RelativePath)
    }
} catch {
    $failure = $_.Exception.Message
} finally {
    $changed = @(foreach ($relative in $protectedFiles) {
        $path = Join-Path $root $relative
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { $relative }
        elseif ((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash -ne $hashesBefore[$relative]) { $relative }
    })
    $remaining = @($targets | Where-Object { Test-Path -LiteralPath $_.FullPath } | ForEach-Object RelativePath)
    $freeAfter = (Get-PSDrive E).Free
    [ordered]@{
        completed=($null -eq $failure -and $remaining.Count -eq 0 -and $changed.Count -eq 0)
        error=$failure; removed=@($removed.ToArray()); remaining=$remaining
        changed_protected_files=$changed; selected_logical_bytes=$total
        free_before_bytes=$freeBefore; free_after_bytes=$freeAfter
        observed_free_change_bytes=($freeAfter-$freeBefore)
    } | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $reportDir 'result.json') -Encoding UTF8
}
Write-Host "Cleanup report: $reportDir"
if ($failure) { throw "Cleanup stopped; see report for partial results. $failure" }
if ($remaining.Count -or $changed.Count) { throw 'Cleanup verification failed; see report.' }
Write-Host ('Completed. E: free space increased by {0:N2} GiB (background disk activity may affect this value).' -f (($freeAfter-$freeBefore)/1GB))
Write-Host 'Current Flow-v3 checkpoints and current environment config hashes are unchanged.'
