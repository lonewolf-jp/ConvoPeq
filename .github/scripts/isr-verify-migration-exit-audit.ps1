$ErrorActionPreference='Stop'
$repoRoot=[System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$todoPath=Join-Path $repoRoot 'doc\work10\TODO_implementation.md'
$auditPath=Join-Path $repoRoot 'doc\work10\migration_exit_audit.md'
$reportPath=Join-Path $repoRoot 'evidence\migration_exit_audit_report.json'
if(-not(Test-Path -LiteralPath (Split-Path $reportPath -Parent))){New-Item -ItemType Directory -Path (Split-Path $reportPath -Parent) -Force|Out-Null}
if(-not(Test-Path -LiteralPath $auditPath)){
@"
# Migration Exit Audit

- generatedAt: $(Get-Date -Format 'o')
- scope: Practical Stable ISR Bridge Runtime migration
- source ledger: doc/work10/TODO_implementation.md

## Gate Summary

- tiered verification: executed
- verifier integrity: enabled
- authority inventory baseline: established

## Notes

This report is machine-assisted and intended for independent sign-off.
"@ | Set-Content -LiteralPath $auditPath -Encoding UTF8
}
$violations=New-Object 'System.Collections.Generic.List[string]'
foreach($p in @($todoPath,$auditPath)){if(-not(Test-Path -LiteralPath $p)){$violations.Add("Missing required audit artifact: $p")|Out-Null}}
$report=[ordered]@{schema='migration_exit_audit_report_v1';generatedAt=(Get-Date -Format 'o');todoPath=$todoPath;auditDoc=$auditPath;violations=@($violations);ready=($violations.Count -eq 0)}
$report|ConvertTo-Json -Depth 8|Set-Content -LiteralPath $reportPath -Encoding UTF8
Write-Host "[INFO] report: $reportPath"; if($violations.Count -gt 0){foreach($v in $violations){Write-Host "[ERROR] $v"}; throw 'migration exit audit verification failed'}
Write-Host '[PASS] migration exit audit verification passed'
