[CmdletBinding()]
param([switch]$Film,[switch]$Unified,[string[]]$Cases)
$ErrorActionPreference='Stop'
$root=(Get-Item -LiteralPath (Join-Path $PSScriptRoot '..')).FullName
$variant=if($Unified){'unified-ipm'}elseif($Film){'fork-film'}else{'spatial'}
$art=Join-Path $root "artifacts/v6-dual-system/$variant"
if(($Unified -or $Film) -and !(Get-Content -LiteralPath (Join-Path $art 'open-loop.json') -Raw | ConvertFrom-Json).passed){throw 'Offline gate failed'}
$proof=Get-Content -LiteralPath (Join-Path $art 'graph.json') -Raw | ConvertFrom-Json
if (($Unified -and !$proof.passed) -or (!$Unified -and !$proof.deploy_graph)) {throw 'HIP Graph verification failed'}
$log=Join-Path $root 'logs/v6-dual-system'
$paths=[ordered]@{}
$port=2200
$allCases=@('fork-left','straight','lane','straight-rolling','fork-right','gentle','curve','cruise','braking','blind')
if ($Unified) {$allCases+=@('follow50','gap-accel','gap-yield')}
if ($Cases) {
    if(@($Cases | Where-Object { $_ -notin $allCases }).Count){throw 'Unregistered scenario'}
    $previous=Get-Content -LiteralPath (Join-Path $art 'closed-loop-runs.json') -Raw | ConvertFrom-Json
    foreach($entry in $previous.PSObject.Properties){$paths[$entry.Name]=$entry.Value}
} else {$Cases=$allCases}
foreach($case in $Cases){
    $town=if($case -in 'curve','braking'){'Town06'}else{'Town04'}
    $scenario=if($case -like 'gap-*'){$case}else{($case -split '-')[0]}
    $opts=@{Town=$town;Scenario=$scenario;Seconds=45;Port=$port;Mode='replay'}
    if ($Unified) {$opts.Unified=$true} else {$opts.Spatial=$true}
    if($Film){$opts.Film=$true}
    if($case -eq 'lane'){$opts.Mode='window-test'}
    if($case -eq 'fork-right'){$opts.ForkRight=$true}
    if($case -eq 'straight-rolling'){$opts.RollingStart=$true}
    $before=@(Get-ChildItem -LiteralPath $log -Directory -Filter "closed-loop-*-$scenario" | ForEach-Object Name)
    & (Join-Path $PSScriptRoot 'Start-FastPilot.ps1') @opts
    $dirs=@(Get-ChildItem -LiteralPath $log -Directory -Filter "closed-loop-*-$scenario" | Where-Object {$_.Name -notin $before})
    if($dirs.Count -ne 1){throw "Ambiguous result for $case"}
    $path=Join-Path $dirs[0].FullName 'summary.json'
    if(!(Test-Path -LiteralPath $path)){throw "Missing $case result"}
    $paths[$case]="logs/v6-dual-system/$($dirs[0].Name)/summary.json"
    $paths | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $art 'closed-loop-runs.json') -Encoding utf8
    $port+=4
}
Push-Location $root
$acceptArgs=if($Film){@('--film')}else{@()}
$module=if($Unified){'v6_dual_system.unified_acceptance'}else{'v6_dual_system.spatial_acceptance'}
try { & '.\.venv-b2d0915\Scripts\python.exe' -m $module @acceptArgs; if($LASTEXITCODE){throw 'Acceptance calculation failed'} } finally {Pop-Location}
