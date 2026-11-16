param([ValidateRange(5,600)][int]$Seconds=120,[ValidateRange(5,40)][int]$SpeedKmh=30)
$ErrorActionPreference='Stop'
$ProjectRoot=Split-Path $PSScriptRoot -Parent
$Python=Join-Path $ProjectRoot '.venv-carla-ue5\Scripts\python.exe'
$Script=Join-Path $PSScriptRoot 'run_occ_closed_loop.py'
try {
    $Current=Invoke-RestMethod -Uri 'http://127.0.0.1:8792/api/status' -TimeoutSec 2
    if ($Current.status -in @('starting','loading_map','settling_vehicles','running')) {
        Write-Output ('OCC session already active: '+$Current.folder)
        return
    }
} catch {}
& (Join-Path $PSScriptRoot 'Start-CarlaUE5.ps1') -MaxFPS 30
$LogRoot=Join-Path $ProjectRoot 'logs\occ-closed-loop'
New-Item -ItemType Directory -Path $LogRoot -Force | Out-Null
$Stamp=Get-Date -Format 'yyyyMMdd-HHmmss'
$Stdout=Join-Path $LogRoot ($Stamp+'-launcher.stdout.log')
$Stderr=Join-Path $LogRoot ($Stamp+'-launcher.stderr.log')
$Arguments=@('-u',('"'+$Script+'"'),'--seconds',$Seconds,'--speed-kmh',$SpeedKmh,'--run-label','live')
# The helper is hidden; CARLA and the local status page are the intended visible UI.
$Process=Start-Process -FilePath $Python -ArgumentList $Arguments -WorkingDirectory $ProjectRoot -WindowStyle Hidden -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr -PassThru
@{pid=$Process.Id;stdout=$Stdout;stderr=$Stderr;requested_seconds=$Seconds;speed_cap_kmh=$SpeedKmh} | ConvertTo-Json | Set-Content (Join-Path $LogRoot 'launcher.json')
Write-Output ('Started OCC controller PID '+$Process.Id+'; status: http://127.0.0.1:8792')
