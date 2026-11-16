param([string]$Subset = '26406,25318,27515', [switch]$Full)
$ErrorActionPreference='Stop'
$env:PYTHONUTF8='1'
$env:PYTHONIOENCODING='utf-8'
$env:PYTHONUNBUFFERED='1'
$R=Split-Path $PSScriptRoot -Parent
$B=Join-Path $R 'runtime\Bench2Drive-0.0.4'
$env:CARLA_ROOT=Join-Path $R 'runtime\CARLA-0.9.15\WindowsNoEditor'
$env:SCENARIO_RUNNER_ROOT=Join-Path $B 'scenario_runner'
$env:PYTHONPATH=(@((Join-Path $env:CARLA_ROOT 'PythonAPI'),(Join-Path $env:CARLA_ROOT 'PythonAPI\carla'),(Join-Path $B 'leaderboard'),(Join-Path $B 'leaderboard\team_code'),$env:SCENARIO_RUNNER_ROOT) -join ';')
$env:IS_BENCH2DRIVE='1'
$env:PLANNER_TYPE='trajectory'
$tag=if ($Full) {'full220'} else {'smoke-'+$Subset.Replace(',','_')}
$out=Join-Path $R ('artifacts\qwen-drive-b2d\'+$tag)
New-Item -ItemType Directory -Force $out | Out-Null
$env:SAVE_PATH=$out
$argsList=@('leaderboard/leaderboard/leaderboard_evaluator.py','--routes','leaderboard/data/bench2drive220.xml','--repetitions','1','--track','SENSORS','--checkpoint',(Join-Path $out 'results.json'),'--debug-checkpoint',(Join-Path $out 'live.txt'),'--agent',(Join-Path $R 'scripts\b2d_official_qwen_drive_agent.py'),'--debug','0','--resume','True','--port','2115','--traffic-manager-port','8115','--gpu-rank','0')
if (!$Full) {$argsList += @('--routes-subset',$Subset)}
Push-Location $B
try { & (Join-Path $R '.venv-b2d0915\Scripts\python.exe') @argsList; if ($LASTEXITCODE -ne 0) {throw "Evaluator failed: $LASTEXITCODE"} } finally {Pop-Location}

