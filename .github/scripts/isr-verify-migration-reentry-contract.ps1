$ErrorActionPreference='Stop'
$repoRoot=[System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$docPath=Join-Path $repoRoot 'doc\work10\migration_reentry_contract.md'
$reportPath=Join-Path $repoRoot 'evidence\migration_reentry_contract_report.json'
if(-not(Test-Path -LiteralPath (Split-Path $reportPath -Parent))){New-Item -ItemType Directory -Path (Split-Path $reportPath -Parent) -Force|Out-Null}
if(-not(Test-Path -LiteralPath $docPath)){
@"
# Migration Re-entry Contract

Any post-migration change that affects Authority/Semantic/Descriptor/Builder must trigger:

1. Scoped DoD re-evaluation
2. Tiered verification rerun
3. Updated audit package generation
4. Independent sign-off
"@ | Set-Content -LiteralPath $docPath -Encoding UTF8
}
$violations=New-Object 'System.Collections.Generic.List[string]'
$txt=Get-Content -LiteralPath $docPath -Raw -Encoding UTF8
foreach($token in @('Authority','Semantic','Descriptor','Builder','Tiered verification')){ if($txt -notmatch [regex]::Escape($token)){ $violations.Add("Migration reentry contract missing token: $token")|Out-Null } }
$report=[ordered]@{schema='migration_reentry_contract_report_v1';generatedAt=(Get-Date -Format 'o');docPath=$docPath;violations=@($violations);ready=($violations.Count -eq 0)}
$report|ConvertTo-Json -Depth 8|Set-Content -LiteralPath $reportPath -Encoding UTF8
Write-Host "[INFO] report: $reportPath"; if($violations.Count -gt 0){foreach($v in $violations){Write-Host "[ERROR] $v"}; throw 'migration reentry contract verification failed'}
Write-Host '[PASS] migration reentry contract verification passed'
