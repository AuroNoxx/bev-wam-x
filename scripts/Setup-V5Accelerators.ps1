$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$uvExe = (Get-Command uv -ErrorAction Stop).Source
& $uvExe pip install --python (Join-Path $projectRoot '.venv-train/Scripts/python.exe') --target (Join-Path $projectRoot 'runtime/v5-qwen-drive-deps') --no-deps --cache-dir (Join-Path $projectRoot '.cache/v5-qwen-drive-world/uv') -r (Join-Path $projectRoot 'configs/v5-qwen-drive-world/requirements-accelerators.txt')
if ($LASTEXITCODE -ne 0) { throw 'Project-local accelerator dependency installation failed.' }
