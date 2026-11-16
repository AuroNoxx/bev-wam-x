param([ValidateSet(30,60)][int]$SimHz=30,[switch]$NoBrowser)
$ErrorActionPreference='Stop'
$Root=Split-Path $PSScriptRoot -Parent
& (Join-Path $PSScriptRoot 'Start-CarlaUE5.ps1') -MaxFPS 60
$Python=Join-Path $Root '.venv-carla-ue5\Scripts\python.exe'
$Ready=$false
for($i=0;$i -lt 30;$i++){
    & $Python -c "import carla;c=carla.Client('127.0.0.1',2100);c.set_timeout(2);print(c.get_server_version())" 2>$null
    if($LASTEXITCODE -eq 0){$Ready=$true;break}
    Start-Sleep -Seconds 2
}
if(!$Ready){throw 'CARLA RPC not ready'}
$Existing=Get-CimInstance Win32_Process | Where-Object {$_.Name -in @('python.exe','pythonw.exe') -and $_.CommandLine -like '*highway-occ-flow*occ_live.py*'}
if(!$Existing){
    New-Item -ItemType Directory -Force (Join-Path $Root 'logs\occ-live') | Out-Null
    $Args=@('-u',('"'+(Join-Path $PSScriptRoot 'occ_live.py')+'"'),'--sim-hz',"$SimHz")
    $Process=Start-Process -FilePath $Python -WorkingDirectory $Root -ArgumentList $Args -WindowStyle Hidden -RedirectStandardOutput (Join-Path $Root 'logs\occ-live\stdout.log') -RedirectStandardError (Join-Path $Root 'logs\occ-live\stderr.log') -PassThru
    $Process.Id | Set-Content (Join-Path $Root 'logs\occ-live\launcher.pid')
}
if(!$NoBrowser){
    Start-Process 'http://127.0.0.1:8765'
    # Only the project CARLA and the browser window showing this dashboard are arranged.
    Start-Process -FilePath $Python -WorkingDirectory $Root -ArgumentList @('"'+(Join-Path $PSScriptRoot 'arrange_viewers.py')+'"') -WindowStyle Hidden | Out-Null
}
Write-Output 'OCC dashboard LEFT / CARLA RIGHT: http://127.0.0.1:8765; stop using scripts/Stop-OccViewer.ps1'
