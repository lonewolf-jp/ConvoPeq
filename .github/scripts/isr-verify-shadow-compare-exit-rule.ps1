$ErrorActionPreference='Stop'
$repoRoot=[System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$cadencePath=Join-Path $repoRoot 'evidence\shadow_compare_cadence.json'
$reportPath=Join-Path $repoRoot 'evidence\shadow_compare_exit_rule_report.json'
if(-not(Test-Path -LiteralPath (Split-Path $reportPath -Parent))){New-Item -ItemType Directory -Path (Split-Path $reportPath -Parent) -Force|Out-Null}
$violations=New-Object 'System.Collections.Generic.List[string]'
if(-not(Test-Path -LiteralPath $cadencePath)){$violations.Add("Missing cadence evidence: $cadencePath")|Out-Null}
$report=[ordered]@{schema='shadow_compare_exit_rule_report_v1';generatedAt=(Get-Date -Format 'o');cadencePath=$cadencePath;violations=@($violations);ready=($violations.Count -eq 0)}
$report|ConvertTo-Json -Depth 8|Set-Content -LiteralPath $reportPath -Encoding UTF8
Write-Host "[INFO] report: $reportPath"; if($violations.Count -gt 0){foreach($v in $violations){Write-Host "[ERROR] $v"}; throw 'shadow compare exit rule verification failed'}
Write-Host '[PASS] shadow compare exit rule verification passed'
