$ErrorActionPreference='Stop'
$repoRoot=[System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$evidencePath=Join-Path $repoRoot 'evidence\retire_timeline.json'
$reportPath=Join-Path $repoRoot 'evidence\retire_monotonicity_report.json'
if(-not(Test-Path -LiteralPath (Split-Path $reportPath -Parent))){New-Item -ItemType Directory -Path (Split-Path $reportPath -Parent) -Force|Out-Null}
$violations=New-Object 'System.Collections.Generic.List[string]'
if(-not(Test-Path -LiteralPath $evidencePath)){$violations.Add("Missing retire timeline evidence: $evidencePath")|Out-Null}
$report=[ordered]@{schema='retire_monotonicity_report_v1';generatedAt=(Get-Date -Format 'o');retireTimeline=$evidencePath;violations=@($violations);ready=($violations.Count -eq 0)}
$report|ConvertTo-Json -Depth 8|Set-Content -LiteralPath $reportPath -Encoding UTF8
Write-Host "[INFO] report: $reportPath"; if($violations.Count -gt 0){ foreach($v in $violations){Write-Host "[ERROR] $v"}; throw 'retire monotonicity verification failed' }
Write-Host '[PASS] retire monotonicity verification passed'
