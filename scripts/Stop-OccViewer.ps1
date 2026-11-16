$ErrorActionPreference='Stop'
# Graceful stop destroys only viewer-owned actors and restores world settings.
Invoke-WebRequest -Uri 'http://127.0.0.1:8765/api/control' -Method Post -ContentType 'application/json' -Body '{"action":"stop"}' | Out-Null
$Clients=Get-CimInstance Win32_Process | Where-Object {$_.Name -eq 'python.exe' -and $_.CommandLine -like '*highway-occ-flow*occ_live.py*'}
if($Clients){$Clients.ProcessId | ForEach-Object {Wait-Process -Id $_ -Timeout 20 -ErrorAction SilentlyContinue}}
Write-Output 'OCC client stop requested; normal cleanup awaited. CARLA GUI is left open.'
