param([int]$Port = 2100, [ValidateRange(20,120)][int]$MaxFPS = 60)
$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path $PSScriptRoot -Parent
$CarlaRoot = Join-Path $ProjectRoot 'runtime\CARLA-1.0\Carla-1.0-Win64-Shipping'
$Exe = Join-Path $CarlaRoot 'CarlaUnreal.exe'
if (!(Test-Path -LiteralPath $Exe)) { throw "Missing UE5 package: $Exe" }
$Existing = Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'CarlaUnreal-Win64-Shipping.exe' -and $_.ExecutablePath -and $_.ExecutablePath.StartsWith($CarlaRoot + '\',[StringComparison]::OrdinalIgnoreCase) }
if ($Existing) { Write-Output 'Project UE5 server already exists; inspect it before starting another.'; return }
# Visible GUI is intentional: the user requested validation of the CARLA simulator.
$Arguments = @('-windowed','-ResX=960','-ResY=540','-quality-level=Low','-dx12',"-carla-rpc-port=$Port",'-nosound',('-ExecCmds="t.MaxFPS {0}"' -f $MaxFPS))
$Process = Start-Process -FilePath $Exe -WorkingDirectory $CarlaRoot -WindowStyle Normal -ArgumentList $Arguments -PassThru
$Process.Id | Set-Content (Join-Path $ProjectRoot 'logs\carla-ue5\launcher.pid')
Write-Output "UE5 GUI launched on RPC port $Port, render cap $MaxFPS FPS (not guaranteed throughput). Synchronous clients may tick more slowly."
