$ErrorActionPreference='Stop'
$repoRoot=[System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$contractPath=Join-Path $repoRoot 'doc\work10\practical_stable_isr_bridge_runtime_complete_migration_plan_2026-05-31.md'
$reportPath=Join-Path $repoRoot 'evidence\shadow_compare_coverage_report.json'
if(-not(Test-Path -LiteralPath (Split-Path $reportPath -Parent))){New-Item -ItemType Directory -Path (Split-Path $reportPath -Parent) -Force|Out-Null}
$violations=New-Object 'System.Collections.Generic.List[string]'
if(Test-Path -LiteralPath $contractPath){$t=Get-Content -LiteralPath $contractPath -Raw -Encoding UTF8; foreach($token in @('Publish','Retire','Crossfade','Observe','Rollback','Recovery')){ if($t -notmatch $token){$violations.Add("Coverage token missing in plan: $token")|Out-Null} }} else {$violations.Add("Missing plan: $contractPath")|Out-Null}
$report=[ordered]@{schema='shadow_compare_coverage_report_v1';generatedAt=(Get-Date -Format 'o');planPath=$contractPath;violations=@($violations);ready=($violations.Count -eq 0)}
$report|ConvertTo-Json -Depth 8|Set-Content -LiteralPath $reportPath -Encoding UTF8
Write-Host "[INFO] report: $reportPath"; if($violations.Count -gt 0){foreach($v in $violations){Write-Host "[ERROR] $v"}; throw 'shadow compare coverage verification failed'}
Write-Host '[PASS] shadow compare coverage verification passed'
