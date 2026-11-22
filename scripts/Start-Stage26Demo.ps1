param(
    [ValidateSet('town04', 'town12')]
    [string]$Route = 'town04',
    [ValidateSet('Low', 'Medium', 'High', 'Epic')]
    [string]$Quality = 'Epic',
    [switch]$Sync,
    [int]$Npc = 0,
    [double]$CruiseKmh = 0,
    [double]$ReplayMinutes = -1,
    [int]$Port = 2460
)

$ErrorActionPreference = 'Stop'
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $RepoRoot

$env:PYTORCH_TUNABLEOP_ENABLED = '1'
$env:PYTORCH_TUNABLEOP_TUNING = '0'
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '3'

if ($Route -eq 'town04') {
    $Plan = 'data/v2-mapless-wa/stage26-colab-20261122/town04-long-ring-plan.json'
    $RouteName = 'town04_full_ring'
    $MaxTicks = 12000
    if ($Npc -le 0) { $Npc = 12 }
    if ($CruiseKmh -le 0) { $CruiseKmh = 55 }
    if ($ReplayMinutes -lt 0) { $ReplayMinutes = 20 }
} else {
    $Plan = 'data/v2-mapless-wa/stage2-nativeC-20261121/town12-plan.json'
    $RouteName = 'town12_val_freeway_exit'
    $MaxTicks = 4200
    if ($Npc -le 0) { $Npc = 12 }
    if ($CruiseKmh -le 0) { $CruiseKmh = 50 }
    if ($ReplayMinutes -lt 0) { $ReplayMinutes = 15 }
}

$ts = Get-Date -Format 'yyyyMMdd-HHmmss'

if ($Sync) {
    $OutDir = "data/v2-mapless-wa/stage26-colab-20261122/demo-$Route-sync-$ts"
    & .\.venv-train\Scripts\python.exe -u -m v2_mapless_wa.phase2_demo `
        --carla-root runtime/Carla-1.0-Win64-Shipping `
        --weights data/v2-mapless-wa/stage26-colab-20261122/stage26.pt `
        --latency data/v2-mapless-wa/stage26-colab-20261122/latency.json `
        --plan $Plan `
        --route $RouteName `
        --max-ticks $MaxTicks `
        --cruise-kmh $CruiseKmh `
        --npc $Npc `
        --npc-along-route `
        --replay-minutes $ReplayMinutes `
        --out $OutDir `
        --port $Port
} else {
    $OutDir = "data/v2-mapless-wa/stage26-colab-20261122/demo-$Route-async-$ts"
    & .\.venv-train\Scripts\python.exe -u -m v2_mapless_wa.phase2_demo_async `
        --carla-root runtime/Carla-1.0-Win64-Shipping `
        --weights data/v2-mapless-wa/stage26-colab-20261122/stage26.pt `
        --latency data/v2-mapless-wa/stage26-colab-20261122/latency.json `
        --plan $Plan `
        --route $RouteName `
        --quality $Quality `
        --max-ticks $MaxTicks `
        --cruise-kmh $CruiseKmh `
        --npc $Npc `
        --npc-along-route `
        --replay-minutes $ReplayMinutes `
        --out $OutDir `
        --port $Port
}
