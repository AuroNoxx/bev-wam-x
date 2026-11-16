# Default: preview only. -CleanSnapCaches removes only Snap3D __pycache__/*.pyc.
# CARLA download cache is inventory-only (no delete switch yet). The earlier note that its
# deletion was "policy-blocked" was Codex Desktop's pre-launch exec policy, not a Windows
# permission or ACL error; a normal PowerShell/shell can delete it.
[CmdletBinding(SupportsShouldProcess = $true)]
param([switch]$CleanSnapCaches)

$ErrorActionPreference = 'Stop'
$snapRoot = 'E:\ai\projects\Snap3D-Coder-4B'
$carlaCache = 'E:\ai\projects\highway-occ-flow\.cache\carla-ue5'

function Assert-RegularPath([string]$Path) {
    $item = Get-Item -LiteralPath $Path -Force
    while ($null -ne $item) {
        if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) {
            throw "Refusing reparse point: $($item.FullName)"
        }
        if ($item -is [IO.DirectoryInfo]) { $item = $item.Parent }
        else { $item = $item.Directory }
    }
}

function Get-RegularFiles([string]$Directory) {
    $pending = [Collections.Generic.Stack[string]]::new()
    $pending.Push($Directory)
    while ($pending.Count -gt 0) {
        foreach ($item in Get-ChildItem -LiteralPath $pending.Pop() -Force) {
            if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) { continue }
            if ($item.PSIsContainer) { $pending.Push($item.FullName) }
            else { $item }
        }
    }
}

Assert-RegularPath $snapRoot
$targets = @(foreach ($name in @('.venv', 'scripts')) {
    $base = Join-Path $snapRoot $name
    if (Test-Path -LiteralPath $base) {
        Assert-RegularPath $base
        Get-RegularFiles $base | Where-Object {
            $_.Extension -eq '.pyc' -and $_.Directory.Name -eq '__pycache__'
        }
    }
})
$selectedBytes = ($targets | Measure-Object Length -Sum).Sum
Write-Host ('Snap3D rebuildable bytecode: {0} files, {1:N2} MiB' -f $targets.Count, ($selectedBytes / 1MB))
if (Test-Path -LiteralPath $carlaCache) {
    Assert-RegularPath $carlaCache
    $carlaBytes = (Get-RegularFiles $carlaCache | Measure-Object Length -Sum).Sum
    Write-Host ('CARLA cache: {0:N2} GiB; inventory only (no delete switch).' -f ($carlaBytes / 1GB))
}
if (-not $CleanSnapCaches) {
    Write-Host 'Preview only. Use -CleanSnapCaches to remove the Snap3D bytecode listed above.'
    return
}
$active = @(Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^python(w)?\.exe$' -and (
        ($_.ExecutablePath -and $_.ExecutablePath.StartsWith($snapRoot + '\', [StringComparison]::OrdinalIgnoreCase)) -or
        ($_.CommandLine -and $_.CommandLine.IndexOf($snapRoot, [StringComparison]::OrdinalIgnoreCase) -ge 0)
    )
})
if ($active.Count) { throw 'Snap3D Python process is active; stop it normally before cleanup.' }
$freeBefore = (Get-PSDrive E).Free
$deleted = 0
foreach ($item in $targets) {
    $resolved = (Resolve-Path -LiteralPath $item.FullName).Path
    if (-not $resolved.StartsWith($snapRoot + '\', [StringComparison]::OrdinalIgnoreCase)) {
        throw "Path outside Snap3D: $resolved"
    }
    Assert-RegularPath $resolved
    if ($PSCmdlet.ShouldProcess($resolved, 'Remove rebuildable Python bytecode')) {
        Remove-Item -LiteralPath $resolved -Force
        $deleted++
    }
}
Write-Host ('Deleted {0} bytecode files. E: free-space change: {1:N2} MiB (includes background disk activity).' -f $deleted, (((Get-PSDrive E).Free - $freeBefore) / 1MB))
