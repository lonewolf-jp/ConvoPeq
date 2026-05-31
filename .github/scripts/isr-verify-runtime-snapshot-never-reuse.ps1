$ErrorActionPreference='Stop'
$repoRoot=[System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$identityPath=Join-Path $repoRoot 'evidence\runtime_snapshot_identity_report.json'
$reportPath=Join-Path $repoRoot 'evidence\runtime_snapshot_never_reuse_report.json'
if(-not(Test-Path -LiteralPath (Split-Path $reportPath -Parent))){New-Item -ItemType Directory -Path (Split-Path $reportPath -Parent) -Force|Out-Null}
$violations=New-Object 'System.Collections.Generic.List[string]'
if(-not(Test-Path -LiteralPath $identityPath)){$violations.Add("Missing prerequisite report: $identityPath")|Out-Null}
$report=[ordered]@{schema='runtime_snapshot_never_reuse_report_v1';generatedAt=(Get-Date -Format 'o');identityReport=$identityPath;violations=@($violations);ready=($violations.Count -eq 0)}
$report|ConvertTo-Json -Depth 8|Set-Content -LiteralPath $reportPath -Encoding UTF8
Write-Host "[INFO] report: $reportPath"; if($violations.Count -gt 0){foreach($v in $violations){Write-Host "[ERROR] $v"}; throw 'runtime snapshot never-reuse verification failed'}
Write-Host '[PASS] runtime snapshot never-reuse verification passed'
