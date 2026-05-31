$ErrorActionPreference='Stop'
$repoRoot=[System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$matrixPath=Join-Path $repoRoot 'doc\work10\matrices\authority_matrix.md'
$reportPath=Join-Path $repoRoot 'evidence\authority_writer_reader_matrix_report.json'
if(-not(Test-Path -LiteralPath (Split-Path $reportPath -Parent))){New-Item -ItemType Directory -Path (Split-Path $reportPath -Parent) -Force|Out-Null}
$violations=New-Object 'System.Collections.Generic.List[string]'
if(Test-Path -LiteralPath $matrixPath){$t=Get-Content -LiteralPath $matrixPath -Raw -Encoding UTF8; foreach($token in @('Writer','Reader','Authority')){ if($t -notmatch [regex]::Escape($token)){ $violations.Add("Authority matrix missing token: $token")|Out-Null } }} else { $violations.Add("Missing matrix: $matrixPath")|Out-Null }
$report=[ordered]@{schema='authority_writer_reader_matrix_report_v1';generatedAt=(Get-Date -Format 'o');matrixPath=$matrixPath;violations=@($violations);ready=($violations.Count -eq 0)}
$report|ConvertTo-Json -Depth 8|Set-Content -LiteralPath $reportPath -Encoding UTF8
Write-Host "[INFO] report: $reportPath"; if($violations.Count -gt 0){foreach($v in $violations){Write-Host "[ERROR] $v"}; throw 'authority writer reader matrix verification failed'}
Write-Host '[PASS] authority writer-reader matrix verification passed'
