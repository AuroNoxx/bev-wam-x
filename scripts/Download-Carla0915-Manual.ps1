# Download only. Reuses completed verified-size chunks. Does not train/start CARLA.
$ErrorActionPreference = 'Stop'
$Root = Split-Path $PSScriptRoot -Parent
$Python = Join-Path $Root 'runtime\python\cpython-3.10.22-windows-x86_64-none\python.exe'
if (!(Test-Path -LiteralPath $Python)) { throw "Project Python not found: $Python" }
Push-Location $Root
try {
    & $Python -u (Join-Path $Root 'scripts\download_carla_0915.py') --connections 4
    if ($LASTEXITCODE -ne 0) { throw 'Download stopped. Completed chunks remain; rerun this script to resume.' }
    $Archive = Join-Path $Root '.cache\carla-0915\CARLA_0.9.15.zip'
    if ((Get-Item -LiteralPath $Archive).Length -ne 7755135185) { throw 'Archive size mismatch' }
    $SevenZip = 'C:\Program Files\7-Zip\7z.exe'
    if (Test-Path -LiteralPath $SevenZip) {
        & $SevenZip t $Archive -bsp0
        if ($LASTEXITCODE -ne 0) { throw 'ZIP integrity check failed; do not extract/run' }
    }
    Write-Host "Download complete: $Archive"
    Write-Host 'No installation, simulation or training has been started.'
} finally { Pop-Location }
