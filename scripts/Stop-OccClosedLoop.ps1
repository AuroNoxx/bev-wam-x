$ErrorActionPreference='Stop'
$ProjectRoot=Split-Path $PSScriptRoot -Parent
$LogRoot=[IO.Path]::GetFullPath((Join-Path $ProjectRoot 'logs\occ-closed-loop'))
$Pointer=Get-Content -LiteralPath (Join-Path $LogRoot 'current.json') -Raw | ConvertFrom-Json
$RunFolder=[IO.Path]::GetFullPath($Pointer.folder)
if (!$RunFolder.StartsWith($LogRoot+'\',[StringComparison]::OrdinalIgnoreCase)) {throw 'Unexpected current run folder'}
Set-Content -LiteralPath (Join-Path $RunFolder 'stop.requested') -Value 'User requested graceful stop'
Write-Output 'Requested controlled stop and cleanup of this run only.'
