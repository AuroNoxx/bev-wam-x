param(
    [string]$Map = 'Town12',
    [ValidateSet('Low', 'Medium', 'High', 'Epic')]
    [string]$Quality = 'Medium',
    [int]$Npc = 35,
    [double]$HighwayKmh = 65,
    [double]$UrbanKmh = 38,
    [int]$MaxTicks = 0,
    [int]$Port = 2460
)

$ErrorActionPreference = 'Stop'
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $RepoRoot

$CompetingCarla = Get-Process -Name 'CarlaUnreal*' -ErrorAction SilentlyContinue
if ($CompetingCarla) {
    $CompetingCarla | Format-Table Name, Id -AutoSize
    throw 'Existing CARLA process detected; stop it before launching Start-Stage32Demo.'
}

$CompetingTrain = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -eq 'python.exe' -and $_.CommandLine -match '(train_stage31_colab|train_stage32|phase2_train|phase26_train|live_carla_demo|stage32_demo_view)'
}
if ($CompetingTrain) {
    $CompetingTrain | Select-Object ProcessId, CommandLine | Format-List
    throw 'Existing training or live demo python process detected.'
}

$env:PYTHONUNBUFFERED = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '3'

Write-Host 'Opening live SR + CARLA viewer...'
& .\.venv-train\Scripts\python.exe -u -m v3_world_model.stage32_demo_view `
    --ckpt .cache/stage32/run-dynamics-v1/latest.ckpt `
    --map $Map `
    --port $Port `
    --quality $Quality `
    --npc $Npc `
    --cruise-highway-kmh $HighwayKmh `
    --cruise-urban-kmh $UrbanKmh `
    --max-ticks $MaxTicks `
    --out-dir logs/stage32-live-demo

if ($LASTEXITCODE -ne 0) { throw "Live demo exited with code $LASTEXITCODE" }
