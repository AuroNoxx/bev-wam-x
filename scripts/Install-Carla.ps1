$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path $PSScriptRoot -Parent
$env:UV_PYTHON_INSTALL_DIR = Join-Path $ProjectRoot 'runtime\python'
$env:UV_CACHE_DIR = Join-Path $ProjectRoot '.cache\uv'
$Downloads = Join-Path $ProjectRoot '.cache\downloads'
$CarlaRoot = Join-Path $ProjectRoot 'runtime\CARLA-0.9.15'
New-Item -ItemType Directory -Force -Path $Downloads | Out-Null
uv python install 3.10.22
if ($LASTEXITCODE -ne 0) { throw 'Python installation failed' }
$LocalPython = Get-ChildItem "$env:UV_PYTHON_INSTALL_DIR\cpython-3.10*-windows-x86_64-none\python.exe" | Select-Object -First 1 -ExpandProperty FullName
if (!(Test-Path -LiteralPath (Join-Path $ProjectRoot '.venv\Scripts\python.exe'))) {
    uv venv --python $LocalPython (Join-Path $ProjectRoot '.venv')
    if ($LASTEXITCODE -ne 0) { throw 'Virtual environment creation failed' }
}
uv pip install --python (Join-Path $ProjectRoot '.venv\Scripts\python.exe') --index-url https://pypi.tuna.tsinghua.edu.cn/simple -r (Join-Path $ProjectRoot 'requirements.txt')
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
if (!(Test-Path -LiteralPath (Join-Path $CarlaRoot 'WindowsNoEditor\CarlaUE4.exe'))) {
    $Archive = Join-Path $Downloads 'CARLA_0.9.15.zip'
    $ExpectedBytes = 7755135185
    if (!(Test-Path -LiteralPath $Archive) -or (Get-Item -LiteralPath $Archive).Length -ne $ExpectedBytes) {
        curl.exe -L --fail --retry 3 --continue-at - --output $Archive 'https://downloads.carlasim.com/Windows/CARLA_0.9.15.zip'
        if ($LASTEXITCODE -ne 0) { throw 'CARLA download failed' }
    }
    if ((Get-Item -LiteralPath $Archive).Length -ne $ExpectedBytes) { throw 'Unexpected archive length; verify official release before proceeding' }
    $SevenZip = 'C:\Program Files\7-Zip\7z.exe'
    if (Test-Path -LiteralPath $SevenZip) {
        & $SevenZip x $Archive "-o$CarlaRoot" -y -bsp0
        if ($LASTEXITCODE -ne 0) { throw 'CARLA extraction failed' }
    } else {
        Expand-Archive -LiteralPath $Archive -DestinationPath $CarlaRoot
    }
}
Write-Output "Installed inside $ProjectRoot. Run scripts\Start-Carla.ps1."
