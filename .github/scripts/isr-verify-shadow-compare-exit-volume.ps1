param([int]$MinObservations=1)
$ErrorActionPreference='Stop'
$repoRoot=[System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$cadencePath=Join-Path $repoRoot 'evidence\shadow_compare_cadence.json'
$reportPath=Join-Path $repoRoot 'evidence\shadow_compare_exit_volume_report.json'
if(-not(Test-Path -LiteralPath (Split-Path $reportPath -Parent))){New-Item -ItemType Directory -Path (Split-Path $reportPath -Parent) -Force|Out-Null}
$violations=New-Object 'System.Collections.Generic.List[string]'
$obs=0
if(Test-Path -LiteralPath $cadencePath){
	$j=Get-Content -LiteralPath $cadencePath -Raw -Encoding UTF8|ConvertFrom-Json
	if($j.PSObject.Properties.Name -contains 'observations'){ $obs=[int]$j.observations }
	elseif($j.PSObject.Properties.Name -contains 'totalObservations'){ $obs=[int]$j.totalObservations }
}else{$violations.Add("Missing cadence evidence: $cadencePath")|Out-Null}
if($obs -lt $MinObservations){$violations.Add("Shadow compare exit volume insufficient: observations=$obs min=$MinObservations")|Out-Null}
$report=[ordered]@{schema='shadow_compare_exit_volume_report_v1';generatedAt=(Get-Date -Format 'o');cadencePath=$cadencePath;observations=$obs;minObservations=$MinObservations;violations=@($violations);ready=($violations.Count -eq 0)}
$report|ConvertTo-Json -Depth 8|Set-Content -LiteralPath $reportPath -Encoding UTF8
Write-Host "[INFO] report: $reportPath"; if($violations.Count -gt 0){foreach($v in $violations){Write-Host "[ERROR] $v"}; throw 'shadow compare exit volume verification failed'}
Write-Host '[PASS] shadow compare exit volume verification passed'
