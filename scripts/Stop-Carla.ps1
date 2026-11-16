$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path $PSScriptRoot -Parent
$ExpectedRoot = Join-Path $ProjectRoot 'runtime\CARLA-0.9.15\'
$ExpectedPython = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
foreach ($Name in @('manual','traffic')) {
    $PidFile = Join-Path $ProjectRoot "logs\$Name.pid"
    if (Test-Path -LiteralPath $PidFile) {
        $SavedProcessId = [int](Get-Content -LiteralPath $PidFile)
        $Process = Get-CimInstance Win32_Process -Filter "ProcessId=$SavedProcessId"
        if ($Process -and $Process.ExecutablePath -in @($ExpectedPython,($ExpectedPython -replace 'python.exe$','pythonw.exe')) -and ($Process.CommandLine -like '*manual_preview.py*' -or $Process.CommandLine -like '*CARLA-0.9.15*')) {
            Stop-Process -Id $SavedProcessId
        }
    }
}
$Servers = @(Get-Process -Name 'CarlaUE4-Win64-Shipping','CarlaUE4' -ErrorAction SilentlyContinue | Where-Object { $_.Path -and $_.Path.StartsWith($ExpectedRoot,[StringComparison]::OrdinalIgnoreCase) })
$Servers | Stop-Process
foreach ($Server in $Servers) {
    if (!$Server.WaitForExit(15000)) { throw "CARLA process $($Server.Id) did not exit" }
}
Get-CimInstance Win32_Process | Where-Object { $_.Name -in @('python.exe','pythonw.exe') -and $_.ExecutablePath -and $_.ExecutablePath.StartsWith($ProjectRoot + '\',[StringComparison]::OrdinalIgnoreCase) -and $_.CommandLine -like '*manual_preview.py*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -ErrorAction SilentlyContinue }
