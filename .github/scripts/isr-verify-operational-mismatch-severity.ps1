$ErrorActionPreference='Stop'
$repoRoot=[System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$cadencePath=Join-Path $repoRoot 'evidence\shadow_compare_cadence.json'
$reportPath=Join-Path $repoRoot 'evidence\operational_mismatch_severity_report.json'
if(-not(Test-Path -LiteralPath (Split-Path $reportPath -Parent))){New-Item -ItemType Directory -Path (Split-Path $reportPath -Parent) -Force|Out-Null}
$violations=New-Object 'System.Collections.Generic.List[string]'
$sevP0=0; $sevP1=0
if(Test-Path -LiteralPath $cadencePath){$j=Get-Content -LiteralPath $cadencePath -Raw -Encoding UTF8|ConvertFrom-Json; if($j.PSObject.Properties.Name -contains 'escalations'){ $sevP1=[int]$j.escalations }}
if($sevP0 -gt 0 -or $sevP1 -gt 0){$violations.Add("Operational mismatch severity threshold violated: p0=$sevP0 p1=$sevP1")|Out-Null}
$report=[ordered]@{schema='operational_mismatch_severity_report_v1';generatedAt=(Get-Date -Format 'o');p0=$sevP0;p1=$sevP1;violations=@($violations);ready=($violations.Count -eq 0)}
$report|ConvertTo-Json -Depth 8|Set-Content -LiteralPath $reportPath -Encoding UTF8
Write-Host "[INFO] report: $reportPath"; if($violations.Count -gt 0){foreach($v in $violations){Write-Host "[ERROR] $v"}; throw 'operational mismatch severity verification failed'}
Write-Host '[PASS] operational mismatch severity verification passed'
