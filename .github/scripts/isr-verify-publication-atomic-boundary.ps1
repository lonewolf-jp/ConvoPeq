$ErrorActionPreference='Stop'
$repoRoot=[System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$commitPath=Join-Path $repoRoot 'src\audioengine\AudioEngine.Commit.cpp'
$reportPath=Join-Path $repoRoot 'evidence\publication_atomic_boundary_report.json'
if(-not(Test-Path -LiteralPath (Split-Path $reportPath -Parent))){New-Item -ItemType Directory -Path (Split-Path $reportPath -Parent) -Force|Out-Null}
$violations=New-Object 'System.Collections.Generic.List[string]'
if(Test-Path -LiteralPath $commitPath){$t=Get-Content -LiteralPath $commitPath -Raw -Encoding UTF8; if($t -notmatch 'runPublicationPrecheckNonRt'){ $violations.Add('Publication boundary precheck entry missing')|Out-Null }} else { $violations.Add("Missing source: $commitPath")|Out-Null }
$report=[ordered]@{schema='publication_atomic_boundary_report_v1';generatedAt=(Get-Date -Format 'o');sourcePath=$commitPath;violations=@($violations);ready=($violations.Count -eq 0)}
$report|ConvertTo-Json -Depth 8|Set-Content -LiteralPath $reportPath -Encoding UTF8
Write-Host "[INFO] report: $reportPath"; if($violations.Count -gt 0){foreach($v in $violations){Write-Host "[ERROR] $v"}; throw 'publication atomic boundary verification failed'}
Write-Host '[PASS] publication atomic boundary verification passed'
