[CmdletBinding()]
param(
    [ValidatePattern('^(Town[0-9]+(HD)?(_Opt)?|Mine_01)$')][string]$Town='Town10HD_Opt',
    [ValidateSet('0.9.15','0.10.0')][string]$SimulatorVersion,
    [ValidateRange(0,40)][int]$NpcCount,
    [ValidateRange(0,120)][int]$InitialSpeed,
    [string]$ModelDirectory,
    [string]$StopFile,
    [ValidateSet('interactive','window-test','replay','gt')][string]$Mode='interactive',
    [ValidateSet('cruise','straight','curve','gentle','comfort','fork','lane','lane-left','lane-right','blind','braking','dagger','city','traffic','traffic-right-fork','follow50','gap-accel','gap-yield')][string]$Scenario='lane',
    [ValidateRange(1,3600)][int]$Seconds=90,
    [ValidateRange(1025,65532)][int]$Port=2176,
    [switch]$Original,
    [switch]$Unified,
    [switch]$LaneSupervised,
    [switch]$RoadROI,
    [switch]$Dagger,
    [switch]$Navigation,
    [switch]$Spatial,
    [switch]$Film,
    [ValidateSet(2,4)][int]$PlanSeconds=4,
    [switch]$ForkRight,
    [switch]$RollingStart,
    [switch]$ReleaseAtCrossing,
    [switch]$ChaseView,
    [switch]$BlueKeep,
    [switch]$BlueCruise,
    [switch]$RecordChase,
    [switch]$CrashEvidence,
    [switch]$BaselineController,
    [long]$TrafficSeed=-1,
    [int]$Seed=30961,
    [int]$CollectFrames=350
)
$ErrorActionPreference='Stop'
$city=if($SimulatorVersion){$SimulatorVersion -eq '0.10.0'}else{$Town -eq 'Town10HD_Opt'}
if($city -and $Mode -eq 'interactive' -and !($Original -or $LaneSupervised -or $RoadROI -or $Dagger -or $Navigation -or $Spatial -or $Film -or $BlueKeep -or $BlueCruise -or $BaselineController)){
    if(!$ModelDirectory){$ModelDirectory=Join-Path $PSScriptRoot '..\models\carla-driver\v1-unified-r22'}
}
if($ModelDirectory){$Unified=$true}
if (!$city -and $Mode -ne 'gt' -and
    !($Original -or $LaneSupervised -or $RoadROI -or $Dagger -or $Navigation -or $Spatial -or $Film -or $BlueKeep -or $BlueCruise -or $BaselineController)) {
    $Unified=$true
}
if ($Unified) {
    if ($Original -or $LaneSupervised -or $RoadROI -or $Dagger -or $Navigation -or $Spatial -or $Film -or $BlueKeep -or $BlueCruise -or $BaselineController -or ($city -and $Mode -ne 'interactive') -or $PlanSeconds -ne 4) {
        throw 'Use one four-second unified checkpoint. Town10 is supported for interactive demonstrations.'
    }
    $art=Join-Path $PSScriptRoot '..\artifacts\v1-dual-system\unified-ipm'
    if($ModelDirectory){$art=(Get-Item -LiteralPath $ModelDirectory -ErrorAction Stop).FullName}
    foreach($file in 'open-loop.json','graph.json') {
        $proof=Join-Path $art $file
        if (!(Test-Path -LiteralPath $proof) -or !(Get-Content -LiteralPath $proof -Raw | ConvertFrom-Json).passed) {
            throw "Unified checkpoint has not passed $file"
        }
    }
}
if ($BlueCruise) {
    if ($PlanSeconds -ne 4) {throw 'Blue cruise uses the full four-second plan'}
    $Film=$true
}
if ($BlueKeep) {
    if ($Original -or $BaselineController -or $Mode -eq 'gt') {throw 'BlueKeep requires a lane-head checkpoint and preview controller.'}
    $ChaseView=$true
}
if ($Mode -eq 'interactive') {$ChaseView=$true}
if ($Film) {$Spatial=$true}
if ($Spatial) {$Navigation=$true}
if ($Navigation -and ($Original -or $Dagger -or $RoadROI -or $LaneSupervised -or $BlueKeep -or $BaselineController -or $Mode -eq 'gt')) {throw 'Choose Navigation alone'}
if ($city) {
    if ($PSBoundParameters.ContainsKey('Scenario') -and $Scenario -ne 'city') {throw 'Choose the city scenario for the UE5 Town10 demo.'}
    if ($Port -gt 65530) {throw 'Town10 traffic also uses Port + 5.'}
    $Scenario='city'
    $ChaseView=$true
}
if ($RecordChase) {$ChaseView=$true}
if ($ChaseView -and $Mode -eq 'gt') {throw 'The model chase view requires a learned driver.'}
if ($Mode -eq 'interactive' -and !($Unified -or $Original -or $LaneSupervised -or $RoadROI -or $Dagger -or $BlueKeep -or $BaselineController -or $Navigation)) {
    $spatialReport=Join-Path $PSScriptRoot '..\artifacts\v1-dual-system\spatial\acceptance.json'
    if ((Test-Path -LiteralPath $spatialReport) -and (Get-Content -LiteralPath $spatialReport -Raw | ConvertFrom-Json).passed) {$Spatial=$true; $Navigation=$true}
    $verified=Join-Path $PSScriptRoot '..\artifacts\v1-dual-system\navigation\acceptance.json'
    if (!$Spatial -and (Test-Path -LiteralPath $verified)) {$Navigation=[bool](Get-Content -LiteralPath $verified -Raw | ConvertFrom-Json).passed}
}
if ($ChaseView -and !($Unified -or $Original -or $LaneSupervised -or $RoadROI -or $Navigation)) {$Dagger=$true}
$project=(Get-Item -LiteralPath (Join-Path $PSScriptRoot '..')).FullName
$sim=Join-Path $project 'runtime\CARLA-0.9.15\WindowsNoEditor'
$python=Join-Path $project '.venv-b2d0915\Scripts\python.exe'
$exe='CarlaUE4.exe'
$child='CarlaUE4\Binaries\Win64\CarlaUE4-Win64-Shipping.exe'
if ($city) {
    $sim=Join-Path $project 'runtime\CARLA-0.10.0\Carla-0.10.0-Win64-Shipping'
    $python=Join-Path $project '.venv-carla-0100\Scripts\python.exe'
    $exe='CarlaUnreal.exe'
    $child='CarlaUnreal\Binaries\Win64\CarlaUnreal-Win64-Shipping.exe'
}
$corrected=Test-Path -LiteralPath (Join-Path $project 'artifacts\v1-dual-system\corrective\completed.json')
$useCorrection=$corrected -and !$Original
$checkpoint=if($useCorrection){'artifacts\v1-dual-system\corrective\best.ckpt'}else{'artifacts\v1-dual-system\best.ckpt'}
if ($Unified) {$checkpoint='artifacts\v1-dual-system\unified-ipm\best.ckpt'}
if ($LaneSupervised) {$checkpoint='artifacts\v1-dual-system\lane-supervised\best.ckpt'}
if ($RoadROI) {$checkpoint='artifacts\v1-dual-system\road-roi\best.ckpt'}
if ($Navigation) {$checkpoint='artifacts\v1-dual-system\navigation\best.ckpt'}
if ($Spatial) {$checkpoint='artifacts\v1-dual-system\spatial\best.ckpt'}
if ($Film) {$checkpoint='artifacts\v1-dual-system\fork-film\best.ckpt'}
if ($Dagger) {$checkpoint='artifacts\v1-dual-system\dagger\best.ckpt'}
$weightPath=if($ModelDirectory){Join-Path $art 'best.ckpt'}else{Join-Path $project $checkpoint}
if ($Mode -ne 'gt' -and !(Test-Path -LiteralPath $weightPath)) { throw "Missing checkpoint $weightPath" }
if ($Mode -eq 'gt' -and !(($Town -eq 'Town04' -and $Scenario -in @('straight','gentle')) -or
                          ($Town -eq 'Town06' -and $Scenario -eq 'curve'))) {throw 'Choose Town04 straight/gentle or Town06 curve for GT'}
if (@(@($LaneSupervised,$Original,$RoadROI,$Dagger) | Where-Object {$_}).Count -gt 1) {throw 'Choose a single checkpoint version'}
if ($RoadROI) {
    Push-Location $project
    try {
        & $python -m v1_dual_system.road_gate --check (Join-Path $project $checkpoint)
        if ($LASTEXITCODE -ne 0) {throw 'Road ROI open-loop gate has not passed'}
    } finally {Pop-Location}
}
$ports=@($Port,($Port+1),($Port+2))
if ($city -or $Scenario -in @('city','traffic','traffic-right-fork')) {$ports+=($Port+5)}
if (Get-NetTCPConnection -State Listen -LocalPort $ports -ErrorAction SilentlyContinue) {
    throw "CARLA ports are in use. Select another -Port to open a separate session."
}
$quality=if ($ChaseView -or $Scenario -eq 'traffic-right-fork') {'Epic'} else {'Low'}
$renderArgs=@(
    '-RenderOffScreen',"-quality-level=$quality",'-dx12',"-carla-rpc-port=$Port",'-nosound','-ResX=640','-ResY=360','-fps=20'
)
if ($city -and $ChaseView) {
    $qualityGroups=@('ViewDistance','AntiAliasing','Shadow','GlobalIllumination','Reflection','PostProcess','Texture','Effects','Foliage','Shading','Landscape')
    $preset=Join-Path $project ".cache\v1-dual-system\ue5-epic-$Port.ini"
    $settings=@('[ScalabilityGroups]','sg.ResolutionQuality=100')+@($qualityGroups|ForEach-Object {"sg.${_}Quality=3"})
    $settings+=@('[/Script/Engine.GameUserSettings]','Version=5','bUseDynamicResolution=False','ResolutionSizeX=1920','ResolutionSizeY=1080','FullscreenMode=2')
    Set-Content -LiteralPath $preset -Value $settings -Encoding ascii
    $commands=@($qualityGroups|ForEach-Object {"sg.${_}Quality 3"})+@('r.ScreenPercentage 100','r.ShadowQuality 5','r.Shadow.Virtual.Enable 1','r.Lumen.DiffuseIndirect.Allow 1','r.Lumen.Reflections.Allow 1')
    $renderArgs=@($renderArgs|Where-Object {$_ -notmatch '^-Res[XY]='})+@('-ResX=1920','-ResY=1080',("-GAMEUSERSETTINGSINI=`"{0}`"" -f $preset),('-ExecCmds="{0}"' -f ($commands -join ',')))
}
$server=Start-Process -FilePath (Join-Path $sim $exe) -WorkingDirectory $sim -WindowStyle Hidden -PassThru -ArgumentList $renderArgs
Push-Location $project
try {
    $ready=$false
    $attempts=if ($city) {240} else {60}
    for ($i=0;$i -lt $attempts;$i++) {
        if($StopFile -and (Test-Path -LiteralPath $StopFile)){return}
        if ($server.HasExited) { throw "CARLA exited before opening its RPC port" }
        if (Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue) {$ready=$true;break}
        Start-Sleep -Milliseconds 500
    }
    if (!$ready) { throw "CARLA did not open its RPC port" }
    $pilot=@('-u','-m','v1_dual_system.run_fast_closed_loop','--town',$Town,'--port',"$Port",'--scenario',$Scenario,'--seconds',"$Seconds")
    $pilot+=@('--simulator-version',$(if($city){'0.10.0'}else{'0.9.15'}))
    if($ModelDirectory){$pilot+=@('--model-directory',$art)}
    if($StopFile){$pilot+=@('--stop-file',$StopFile)}
    if($PSBoundParameters.ContainsKey('NpcCount')){$pilot+=@('--npc-count',"$NpcCount")}
    if($PSBoundParameters.ContainsKey('InitialSpeed')){$pilot+=@('--initial-target',"$InitialSpeed")}
    if($TrafficSeed -ge 0){$pilot+=@('--traffic-seed',"$TrafficSeed")}
    $pilot+=@('--plan-seconds',"$PlanSeconds")
    $pilot+=@('--seed',"$Seed",'--collect-frames',"$CollectFrames")
    if ($Unified) {$pilot+='--unified'} elseif ($Film) {$pilot+='--film'} elseif ($Spatial) {$pilot+='--spatial'} elseif ($Navigation) {$pilot+='--navigation'} elseif ($Dagger) {$pilot+='--dagger'} elseif ($RoadROI) {$pilot+='--road-roi'} elseif ($LaneSupervised) {$pilot+='--lane-supervised'} elseif ($useCorrection) {$pilot+='--corrective'}
    if ($ForkRight) {$pilot+='--fork-right'}
    if ($RollingStart) {$pilot+='--rolling-start'}
    if ($ReleaseAtCrossing) {$pilot+='--release-at-crossing'}
    if ($ChaseView) {$pilot+='--chase-view'}
    if ($BlueKeep) {$pilot+='--blue-keep'}
    if ($BlueCruise) {$pilot+='--blue-cruise'}
    if ($RecordChase) {$pilot+='--record-chase'}
    if ($CrashEvidence) {$pilot+='--crash-evidence'}
    if ($BaselineController) {$pilot+='--baseline-controller'}
    if ($Mode -eq 'interactive') {$pilot+='--interactive'}
    if ($Mode -eq 'window-test') {$pilot+='--keyboard-test'}
    if ($Mode -eq 'gt') {
        $pilot=@('-u','-m','v1_dual_system.diagnose_gt_controller','--town',$Town,'--port',"$Port",'--seconds',"$Seconds")
        if (!$BaselineController) {$pilot+='--preview'}
        if ($Scenario -eq 'gentle') {$pilot+='--gentle'}
    }
    & $python @pilot
    if ($LASTEXITCODE -ne 0) {throw "Pilot exited with code $LASTEXITCODE"}
} finally {
    Pop-Location
    $childExe=Join-Path $sim $child
    Get-CimInstance Win32_Process -Filter "ParentProcessId=$($server.Id)" |
        Where-Object { $_.ExecutablePath -eq $childExe -and $_.CommandLine -like "*-carla-rpc-port=$Port *" } |
        ForEach-Object { Stop-Process -Id $_.ProcessId -ErrorAction SilentlyContinue }
    if (!$server.WaitForExit(1500)) { $server.Kill() }
}
