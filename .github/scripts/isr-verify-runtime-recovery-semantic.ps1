$ErrorActionPreference='Stop'
$repoRoot=[System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$contractPath=Join-Path $repoRoot 'doc\work10\contracts\runtime_recovery_semantic.md'
$reportPath=Join-Path $repoRoot 'evidence\runtime_recovery_semantic_report.json'
if(-not(Test-Path -LiteralPath (Split-Path $reportPath -Parent))){New-Item -ItemType Directory -Path (Split-Path $reportPath -Parent) -Force|Out-Null}
$violations=New-Object 'System.Collections.Generic.List[string]'
if(Test-Path -LiteralPath $contractPath){$t=Get-Content -LiteralPath $contractPath -Raw -Encoding UTF8; foreach($token in @('Recoverable','Retryable','Fatal')){ if($t -notmatch [regex]::Escape($token)){ $violations.Add("Recovery semantic token missing: $token")|Out-Null } }} else { $violations.Add("Missing contract: $contractPath")|Out-Null }
$report=[ordered]@{schema='runtime_recovery_semantic_report_v1';generatedAt=(Get-Date -Format 'o');contractPath=$contractPath;violations=@($violations);ready=($violations.Count -eq 0)}
$report|ConvertTo-Json -Depth 8|Set-Content -LiteralPath $reportPath -Encoding UTF8
Write-Host "[INFO] report: $reportPath"; if($violations.Count -gt 0){foreach($v in $violations){Write-Host "[ERROR] $v"}; throw 'runtime recovery semantic verification failed'}
Write-Host '[PASS] runtime recovery semantic verification passed'
