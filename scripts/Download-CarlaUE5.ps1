param(
    [ValidateSet(4,8,16)][int]$Connections = 16,
    [switch]$DryRun
)
$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path $PSScriptRoot -Parent
$Python = Join-Path $ProjectRoot '.venv-train\Scripts\python.exe'
$Archive = Join-Path $ProjectRoot '.cache\carla-ue5\CARLA_1.0.zip'
if (!(Test-Path -LiteralPath $Python)) { throw "Project Python missing: $Python" }
if (!(Get-Command curl.exe -ErrorAction SilentlyContinue)) { throw 'curl.exe is required' }
New-Item -ItemType Directory -Force -Path (Join-Path $ProjectRoot 'logs\carla-ue5') | Out-Null
$DownloadArgs = @('-u',(Join-Path $PSScriptRoot 'download_carla_ue5.py'),'--connections',"$Connections")
if ($DryRun) { $DownloadArgs += '--dry-run' }
& $Python @DownloadArgs
if ($LASTEXITCODE -ne 0) { throw 'Download stopped or failed; rerun this command to reuse completed chunks.' }
if ($DryRun) { return }
Write-Output 'Checking complete ZIP integrity...'
$SevenZip = 'C:\Program Files\7-Zip\7z.exe'
if (Test-Path -LiteralPath $SevenZip) {
    & $SevenZip t $Archive -bsp0
} else {
    & $Python -c 'import sys,zipfile; z=zipfile.ZipFile(sys.argv[1]); bad=z.testzip(); print("ZIP integrity:",bad or "OK"); sys.exit(1 if bad else 0)' $Archive
}
if ($LASTEXITCODE -ne 0) { throw 'ZIP integrity FAILED. Do not extract/run; keep chunks for diagnosis.' }
$Hash = Get-FileHash -LiteralPath $Archive -Algorithm SHA256
$Hash.Hash | Set-Content (Join-Path $ProjectRoot 'logs\carla-ue5\archive-sha256.txt')
Write-Output "Download and ZIP verification complete: $Archive"
Write-Output "Local SHA256 (not an official signature): $($Hash.Hash)"
Write-Output 'Nothing was installed, launched, or deleted. Keep the chunks until installation is verified.'
