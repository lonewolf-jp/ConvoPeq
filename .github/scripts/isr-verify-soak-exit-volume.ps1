param([int]$MinEvents=1)
$ErrorActionPreference='Stop'
$repoRoot=[System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$cadencePath=Join-Path $repoRoot 'evidence\shadow_compare_cadence.json'
$reportPath=Join-Path $repoRoot 'evidence\soak_exit_volume_report.json'
if(-not(Test-Path -LiteralPath (Split-Path $reportPath -Parent))){New-Item -ItemType Directory -Path (Split-Path $reportPath -Parent) -Force|Out-Null}
$violations=New-Object 'System.Collections.Generic.List[string]'
$observations=0
if(Test-Path -LiteralPath $cadencePath){
	$j=Get-Content -LiteralPath $cadencePath -Raw -Encoding UTF8|ConvertFrom-Json
	if($j.PSObject.Properties.Name -contains 'observations'){ $observations=[int]$j.observations }
	elseif($j.PSObject.Properties.Name -contains 'totalObservations'){ $observations=[int]$j.totalObservations }
}
if($observations -lt $MinEvents){$violations.Add("Soak exit volume below minimum: observations=$observations min=$MinEvents")|Out-Null}
$report=[ordered]@{schema='soak_exit_volume_report_v1';generatedAt=(Get-Date -Format 'o');cadencePath=$cadencePath;observations=$observations;minEvents=$MinEvents;violations=@($violations);ready=($violations.Count -eq 0)}
$report|ConvertTo-Json -Depth 8|Set-Content -LiteralPath $reportPath -Encoding UTF8
Write-Host "[INFO] report: $reportPath"; if($violations.Count -gt 0){foreach($v in $violations){Write-Host "[ERROR] $v"}; throw 'soak exit volume verification failed'}
Write-Host '[PASS] soak exit volume verification passed'
