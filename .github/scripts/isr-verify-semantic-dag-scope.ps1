$ErrorActionPreference='Stop'
$repoRoot=[System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$todoPath=Join-Path $repoRoot 'doc\work10\TODO_implementation.md'
$reportPath=Join-Path $repoRoot 'evidence\semantic_dag_scope_report.json'
if(-not(Test-Path -LiteralPath (Split-Path $reportPath -Parent))){New-Item -ItemType Directory -Path (Split-Path $reportPath -Parent) -Force|Out-Null}
$violations=New-Object 'System.Collections.Generic.List[string]'
if(Test-Path -LiteralPath $todoPath){$t=Get-Content -LiteralPath $todoPath -Raw -Encoding UTF8; if($t -notmatch 'semantic DAG specification'){ $violations.Add('TODO ledger missing semantic DAG specification scope entry')|Out-Null }} else { $violations.Add("Missing TODO: $todoPath")|Out-Null }
$report=[ordered]@{schema='semantic_dag_scope_report_v1';generatedAt=(Get-Date -Format 'o');todoPath=$todoPath;violations=@($violations);ready=($violations.Count -eq 0)}
$report|ConvertTo-Json -Depth 8|Set-Content -LiteralPath $reportPath -Encoding UTF8
Write-Host "[INFO] report: $reportPath"; if($violations.Count -gt 0){foreach($v in $violations){Write-Host "[ERROR] $v"}; throw 'semantic dag scope verification failed'}
Write-Host '[PASS] semantic dag scope verification passed'
