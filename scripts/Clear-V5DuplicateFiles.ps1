[CmdletBinding()]
param([switch]$Apply)
$ErrorActionPreference='Stop'
$root=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..')).TrimEnd('\')
$audit=Get-Content -LiteralPath "$root/artifacts/v5-qwen-drive-world/duplicate-occ-audit.json" -Raw | ConvertFrom-Json
if($audit.phase -ne 'verified'){throw 'Archive comparison has not passed'}
function Check-LocalPath([string]$path) {
    $full=[IO.Path]::GetFullPath($path)
    if(-not $full.StartsWith($root+'\',[StringComparison]::OrdinalIgnoreCase)){throw "Outside project: $full"}
    $cursor=$full
    while($cursor.Length -ge $root.Length){
        $item=Get-Item -LiteralPath $cursor -Force
        if($item.Attributes -band [IO.FileAttributes]::ReparsePoint){throw "Reparse path: $cursor"}
        if($cursor -eq $root){break}
        $cursor=[IO.Path]::GetDirectoryName($cursor)
    }
    return $full
}
$readers=@(Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(w)?\.exe$' -and $_.CommandLine -match 'highway-occ-flow|v5_qwen_drive_world|train_lightning' })
if($readers.Count){throw 'A project Python task is still running; no files removed'}
foreach($a in $audit.archives){
    $path=Check-LocalPath $a.path
    if((Get-Item -LiteralPath $path).Length -ne $a.bytes){throw "Archive changed: $path"}
}
foreach($parent in ($audit.records.path | ForEach-Object {Split-Path -Parent $_} | Sort-Object -Unique)){[void](Check-LocalPath $parent)}
$candidates=@()
foreach($r in $audit.records){
    $path=[IO.Path]::GetFullPath($r.path)
    if(-not $path.StartsWith($root+'\data\public\bench2drive\clips\',[StringComparison]::OrdinalIgnoreCase) -or $path -notmatch '\\occ\\[^\\]+\.npz$'){throw 'Unexpected candidate'}
    if(-not (Test-Path -LiteralPath $path)){continue}
    $item=Get-Item -LiteralPath $path
    $stamp=($item.LastWriteTimeUtc.Ticks-[datetime]::UnixEpoch.Ticks)*100
    if($item.Length -ne $r.bytes -or $stamp -ne $r.mtime_ns){throw "Verified copy changed: $path"}
    $candidates+=[pscustomobject]@{path=$path;bytes=$item.Length;reason='byte-identical obsolete OCC extraction; original archive retained'}
}
foreach($item in Get-ChildItem -LiteralPath "$root/artifacts" -Recurse -File -Filter 'atomic.pending'){
    [void](Check-LocalPath $item.FullName)
    $candidates+=[pscustomobject]@{path=$item.FullName;bytes=$item.Length;reason='old atomic-save failure-test temporary; not latest/best'}
}
$before=(Get-PSDrive -Name ([IO.Path]::GetPathRoot($root).Substring(0,1))).Free
$record=[ordered]@{phase='preview';files=$candidates;bytes=($candidates.bytes | Measure-Object -Sum).Sum;free_before=$before;removed=0;error=$null}
$out="$root/artifacts/v5-qwen-drive-world/duplicate-cleanup-result.json"
$record | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $out
if(-not $Apply){Write-Output "Preview: $($candidates.Count) files, $([math]::Round($record.bytes/1GB,3)) GiB";exit}
try {
    foreach($item in $candidates){Remove-Item -LiteralPath $item.path -Force;$record.removed++}
    $record.phase='completed'
} catch {$record.phase='failed';$record.error=$_.Exception.Message;throw}
finally {
    $record.free_after=(Get-PSDrive -Name ([IO.Path]::GetPathRoot($root).Substring(0,1))).Free
    $record.actual_free_delta=$record.free_after-$before
    $record | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $out
}
Write-Output "Removed $($record.removed) files; $([math]::Round($record.bytes/1GB,3)) GiB logical, $([math]::Round($record.actual_free_delta/1GB,3)) GiB disk increase"
