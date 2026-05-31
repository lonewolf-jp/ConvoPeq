$ErrorActionPreference='Stop'
$repoRoot=[System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$contractPath=Join-Path $repoRoot 'doc\work10\contracts\runtime_semantic_transition_graph.md'
$reportPath=Join-Path $repoRoot 'evidence\semantic_dag_report.json'
if(-not(Test-Path -LiteralPath (Split-Path $reportPath -Parent))){New-Item -ItemType Directory -Path (Split-Path $reportPath -Parent) -Force|Out-Null}
$violations=New-Object 'System.Collections.Generic.List[string]'
if(Test-Path -LiteralPath $contractPath){$t=Get-Content -LiteralPath $contractPath -Raw -Encoding UTF8; foreach($token in @('Nodes','Edge Rules','Invalid Patterns')){ if($t -notmatch [regex]::Escape($token)){ $violations.Add("Semantic DAG contract missing token: $token")|Out-Null } }} else { $violations.Add("Missing contract: $contractPath")|Out-Null }
$report=[ordered]@{schema='semantic_dag_report_v1';generatedAt=(Get-Date -Format 'o');contractPath=$contractPath;violations=@($violations);ready=($violations.Count -eq 0)}
$report|ConvertTo-Json -Depth 8|Set-Content -LiteralPath $reportPath -Encoding UTF8
Write-Host "[INFO] report: $reportPath"; if($violations.Count -gt 0){foreach($v in $violations){Write-Host "[ERROR] $v"}; throw 'semantic dag verification failed'}
Write-Host '[PASS] semantic dag verification passed'
