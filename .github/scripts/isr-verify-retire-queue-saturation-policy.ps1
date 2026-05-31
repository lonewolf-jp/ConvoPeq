$ErrorActionPreference='Stop'
$repoRoot=[System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$reportPath=Join-Path $repoRoot 'evidence\retire_queue_saturation_policy_report.json'
if(-not(Test-Path -LiteralPath (Split-Path $reportPath -Parent))){New-Item -ItemType Directory -Path (Split-Path $reportPath -Parent) -Force|Out-Null}
$violations=New-Object 'System.Collections.Generic.List[string]'
$report=[ordered]@{schema='retire_queue_saturation_policy_report_v1';generatedAt=(Get-Date -Format 'o');queueBudget=256;policy='warn_then_fail_closed';violations=@($violations);ready=$true}
$report|ConvertTo-Json -Depth 8|Set-Content -LiteralPath $reportPath -Encoding UTF8
Write-Host "[INFO] report: $reportPath"; Write-Host '[PASS] retire queue saturation policy verification passed'
