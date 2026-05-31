$ErrorActionPreference='Stop'
$repoRoot=[System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$reportPath=Join-Path $repoRoot 'evidence\exit_audit_independence_report.json'
$governancePath=Join-Path $repoRoot 'doc\work10\ai_implementation_governance_v2_1.md'
if(-not(Test-Path -LiteralPath (Split-Path $reportPath -Parent))){New-Item -ItemType Directory -Path (Split-Path $reportPath -Parent) -Force|Out-Null}
$violations=New-Object 'System.Collections.Generic.List[string]'
if(Test-Path -LiteralPath $governancePath){
	$t=Get-Content -LiteralPath $governancePath -Raw -Encoding UTF8
	if(($t -notmatch 'Rule-5') -and ($t -notmatch 'independ')){ $violations.Add('Governance document missing independence rule')|Out-Null }
} else { $violations.Add("Missing governance doc: $governancePath")|Out-Null }
$report=[ordered]@{schema='exit_audit_independence_report_v1';generatedAt=(Get-Date -Format 'o');governancePath=$governancePath;violations=@($violations);ready=($violations.Count -eq 0)}
$report|ConvertTo-Json -Depth 8|Set-Content -LiteralPath $reportPath -Encoding UTF8
Write-Host "[INFO] report: $reportPath"; if($violations.Count -gt 0){foreach($v in $violations){Write-Host "[ERROR] $v"}; throw 'exit audit independence verification failed'}
Write-Host '[PASS] exit audit independence verification passed'
