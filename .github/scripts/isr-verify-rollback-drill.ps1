$ErrorActionPreference='Stop'
$repoRoot=[System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$rollbackMatrixPath=Join-Path $repoRoot 'evidence\rollback_compatibility_report.json'
$reportPath=Join-Path $repoRoot 'evidence\rollback_drill_report.json'
if(-not(Test-Path -LiteralPath (Split-Path $reportPath -Parent))){New-Item -ItemType Directory -Path (Split-Path $reportPath -Parent) -Force|Out-Null}
$violations=New-Object 'System.Collections.Generic.List[string]'
if(-not(Test-Path -LiteralPath $rollbackMatrixPath)){$violations.Add("Missing rollback matrix evidence: $rollbackMatrixPath")|Out-Null}
$report=[ordered]@{schema='rollback_drill_report_v1';generatedAt=(Get-Date -Format 'o');rollbackMatrixPath=$rollbackMatrixPath;violations=@($violations);ready=($violations.Count -eq 0)}
$report|ConvertTo-Json -Depth 8|Set-Content -LiteralPath $reportPath -Encoding UTF8
Write-Host "[INFO] report: $reportPath"; if($violations.Count -gt 0){foreach($v in $violations){Write-Host "[ERROR] $v"}; throw 'rollback drill verification failed'}
Write-Host '[PASS] rollback drill verification passed'
