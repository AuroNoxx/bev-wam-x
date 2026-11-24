param(
    [string]$Map = 'Town12',
    [ValidateSet('Low', 'Medium', 'High', 'Epic')]
    [string]$Quality = 'Medium',
    [int]$Npc = 35,
    [ValidateSet('raw', 'clean')]
    [string]$SrMode = 'raw',
    [int]$MaxTicks = 0,
    [int]$Port = 2460,
    [string]$Ckpt = 'models/v3_stage31b_step7968/best.ckpt'
)

$ErrorActionPreference = 'Stop'
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $RepoRoot

$CompetingCarla = Get-Process -Name 'CarlaUnreal*' -ErrorAction SilentlyContinue
if ($CompetingCarla) {
    $CompetingCarla | Format-Table Name, Id -AutoSize
    throw 'Existing CARLA process detected; stop it before launching Start-Stage31Demo.'
}

$CompetingTrain = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -eq 'python.exe' -and $_.CommandLine -match '(train_stage31_colab|train_stage32|phase2_train|phase26_train|live_carla_demo|stage32_demo_view|live_perception_demo)'
}
if ($CompetingTrain) {
    $CompetingTrain | Select-Object ProcessId, CommandLine | Format-List
    throw 'Existing training or live demo python process detected.'
}

$env:PYTHONUNBUFFERED = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '3'

Write-Host "Opening Stage 3.1 Perception Audit viewer (TM Autopilot, map=$Map, sr_mode=$SrMode)..."
& .\.venv-train\Scripts\python.exe -u -m v3_world_model.live_perception_demo `
    --ckpt $Ckpt `
    --map $Map `
    --port $Port `
    --quality $Quality `
    --npc $Npc `
    --sr-mode $SrMode `
    --max-ticks $MaxTicks `
    --out-dir logs/stage31-perception-demo

if ($LASTEXITCODE -ne 0) { throw "Stage 3.1 perception audit exited with code $LASTEXITCODE" }
