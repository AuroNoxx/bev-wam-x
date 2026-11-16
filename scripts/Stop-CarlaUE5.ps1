$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path $PSScriptRoot -Parent
$CarlaRoot = [IO.Path]::GetFullPath((Join-Path $ProjectRoot 'runtime\CARLA-1.0'))
$Processes = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -in @('CarlaUnreal.exe','CarlaUnreal-Win64-Shipping.exe') -and
    $_.ExecutablePath -and $_.ExecutablePath.StartsWith($CarlaRoot + '\',[StringComparison]::OrdinalIgnoreCase)
}
foreach ($Process in $Processes) { Stop-Process -Id $Process.ProcessId -ErrorAction SilentlyContinue }
Write-Output "Stopped only project-local UE5 processes: $($Processes.ProcessId -join ', ')"
