param([switch]$ServerOnly, [switch]$Manual)
$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path $PSScriptRoot -Parent
$CarlaRoot = Join-Path $ProjectRoot 'runtime\CARLA-0.9.15\WindowsNoEditor'
$Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$LogDir = Join-Path $ProjectRoot 'logs'
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$ServerExe = Join-Path $CarlaRoot 'CarlaUE4.exe'
if (!(Test-Path -LiteralPath $ServerExe)) { throw "Missing CARLA executable: $ServerExe" }
$Existing = Get-Process -Name 'CarlaUE4-Win64-Shipping' -ErrorAction SilentlyContinue | Where-Object { $_.Path -and $_.Path.StartsWith($CarlaRoot + '\',[StringComparison]::OrdinalIgnoreCase) }
if (!$Existing) {
    # Visible simulator window is intentional; the user requested a GUI preview.
    Start-Process -FilePath $ServerExe -WorkingDirectory $CarlaRoot -WindowStyle Normal -ArgumentList '-windowed','-ResX=1280','-ResY=720','-quality-level=Low','-dx12','-carla-rpc-port=2000','-nosound' | Out-Null
}
if ($ServerOnly) { return }
$Ready = $false
for ($Attempt = 0; $Attempt -lt 30; $Attempt++) {
    & $Python -c "import carla; c=carla.Client('127.0.0.1',2000); c.set_timeout(2); print(c.get_server_version())" 2>$null
    if ($LASTEXITCODE -eq 0) { $Ready = $true; break }
    Start-Sleep -Seconds 2
}
if (!$Ready) { throw 'CARLA RPC was not ready. Check the simulator window and Saved\Logs.' }
$ExistingClient = Get-CimInstance Win32_Process | Where-Object { $_.Name -in @('python.exe','pythonw.exe') -and $_.CommandLine -like "*$ProjectRoot*manual_preview.py*" }
if ($ExistingClient) { Write-Output 'The driving client is already running. Restore its pygame window from the taskbar.'; return }
$MapSetup = "import carla; c=carla.Client('127.0.0.1',2000); c.set_timeout(60); w=c.get_world(); w=c.load_world('Town04') if not w.get_map().name.endswith('/Town04') else w; w.set_weather(carla.WeatherParameters.ClearNoon); print(w.get_map().name)"
& $Python -c $MapSetup
if ($LASTEXITCODE -ne 0) { throw 'Failed to prepare Town04' }
$ManualScript = Join-Path $PSScriptRoot 'manual_preview.py'
$PythonWindowed = Join-Path $ProjectRoot '.venv\Scripts\pythonw.exe'
# This is the user-visible driving GUI, not a hidden background helper.
$ClientArgs = @('-u',$ManualScript,'--res','1280x720','--filter','vehicle.tesla.model3','--sync')
if (!$Manual) { $ClientArgs += '--autopilot' }
$ClientProcess = Start-Process -FilePath $PythonWindowed -WorkingDirectory (Split-Path $ManualScript) -ArgumentList $ClientArgs -WindowStyle Normal -RedirectStandardOutput (Join-Path $LogDir 'manual.stdout.log') -RedirectStandardError (Join-Path $LogDir 'manual.stderr.log') -PassThru
$ClientProcess.Id | Set-Content (Join-Path $LogDir 'manual.pid')
Write-Output 'CARLA preview started. P: autopilot/manual; WASD: drive; Tab: camera; Esc: close driving window.'
