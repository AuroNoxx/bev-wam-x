$ErrorActionPreference='Stop'
$root=[IO.Path]::GetFullPath('E:\ai\projects\highway-occ-flow')
$j=Get-Content -LiteralPath "$root\artifacts\implementation-plan-v2\proposed_weight_cleanup_exact.json" -Raw|ConvertFrom-Json
if($j.proposed_delete_files.Count -ne 61){throw 'Manifest count changed'}
$processes=Get-CimInstance Win32_Process|Where-Object {$_.Name -match 'python'}
foreach($p in $processes){if($p.CommandLine -notmatch 'tensorboard.main|train_structural_prototypes_light512.py'){throw 'Unexpected Python process; stop cleanup'}}
$seen=@{}
foreach($r in @($j.proposed_delete_files)+@($j.retain_candidates)){
 $p=[IO.Path]::GetFullPath($r.absolute_path)
 if(-not $p.StartsWith($root+'\artifacts\',[StringComparison]::OrdinalIgnoreCase)){throw "Outside artifacts: $p"}
 if($seen.ContainsKey($p)){throw "Duplicate/overlap $p"};$seen[$p]=$true
 $f=Get-Item -LiteralPath $p
 if($f.PSIsContainer -or ($f.Attributes -band [IO.FileAttributes]::ReparsePoint)){throw "Unexpected file type $p"}
 $ns=[decimal]($f.LastWriteTimeUtc.Ticks-621355968000000000)*100
 if($f.Length -ne $r.bytes -or $ns -ne [decimal]$r.last_write_ns){throw "Metadata mismatch: $p"}
 if($p -match 'structural-prototypes'){throw 'New experiment in cleanup manifest'}
}
$before=(Get-PSDrive E).Free
$deleted=@();$errors=@()
foreach($r in $j.proposed_delete_files){
 try {
  $p=[IO.Path]::GetFullPath($r.absolute_path);$f=Get-Item -LiteralPath $p
  if($f.Length -ne $r.bytes -or [decimal]($f.LastWriteTimeUtc.Ticks-621355968000000000)*100 -ne [decimal]$r.last_write_ns){throw 'Changed since preflight'}
  $handle=[IO.File]::Open($p,[IO.FileMode]::Open,[IO.FileAccess]::Read,[IO.FileShare]::None);$handle.Dispose()
  Remove-Item -LiteralPath $p -Force
  if(Test-Path -LiteralPath $p){throw 'Still exists'}
  $deleted+=@{path=$p;bytes=$r.bytes}
 } catch {$errors+=@{path=$r.absolute_path;error=$_.Exception.Message}}
}
$retained=@($j.retain_candidates|Where-Object {Test-Path -LiteralPath $_.absolute_path}).Count
$after=(Get-PSDrive E).Free
$result=@{user_authorization='Parent forwarded owner explicit permanent clear approval for exact 61-file manifest';timestamp_utc=[datetime]::UtcNow.ToString('o');deleted_count=$deleted.Count;deleted_bytes=($deleted|Measure-Object bytes -Sum).Sum;retained_major_count=$retained;free_bytes_before=$before;free_bytes_after=$after;free_bytes_change=$after-$before;errors=$errors;deleted=$deleted;active_run_untouched='structural-prototypes-light512';scope='Files only; no directory recursion; docs/configs/logs untouched'}
$result|ConvertTo-Json -Depth 6|Set-Content -LiteralPath "$root\artifacts\implementation-plan-v2\weight_cleanup_execution.json" -Encoding utf8
$result|Select-Object deleted_count,deleted_bytes,retained_major_count,free_bytes_change,errors|ConvertTo-Json -Depth 4
