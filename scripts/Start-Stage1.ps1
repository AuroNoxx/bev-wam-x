param([switch]$Resume,[switch]$FinishMaxEpochs)
$ErrorActionPreference='Stop'
$Root=Split-Path $PSScriptRoot -Parent
$Python=Join-Path $Root '.venv-train\Scripts\python.exe'
$Log=Join-Path $Root 'logs/stage1-v3'
New-Item -ItemType Directory -Path $Log -Force | Out-Null
if(Get-CimInstance Win32_Process | Where-Object {$_.Name -eq 'python.exe' -and $_.CommandLine -like '*highway-occ-flow*run_stage1.py*'}){Write-Output 'Stage 1 launcher already running';return}
$Arguments=@('-u',('"'+(Join-Path $PSScriptRoot 'run_stage1.py')+'"'))
if($Resume){$Arguments+='--resume'}
if($FinishMaxEpochs){$Arguments+='--finish-max-epochs'}
$P=Start-Process $Python -ArgumentList $Arguments -WorkingDirectory $Root -WindowStyle Hidden -RedirectStandardOutput (Join-Path $Log 'launcher.stdout.log') -RedirectStandardError (Join-Path $Log 'launcher.stderr.log') -PassThru
$P.Id | Set-Content (Join-Path $Log 'stage1.pid')
Write-Output "Stage 1 launcher PID $($P.Id); state: logs/stage1-v3/stage1-launch.json"
