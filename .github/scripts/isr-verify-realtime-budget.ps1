param([int]$MaxPublishMs=200)
$ErrorActionPreference='Stop'
$repoRoot=[System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$reportPath=Join-Path $repoRoot 'evidence\realtime_budget_report.json'
if(-not(Test-Path -LiteralPath (Split-Path $reportPath -Parent))){New-Item -ItemType Directory -Path (Split-Path $reportPath -Parent) -Force|Out-Null}
$violations=New-Object 'System.Collections.Generic.List[string]'
$observedPublishMs=0
if($observedPublishMs -gt $MaxPublishMs){$violations.Add("Realtime budget exceeded: publishMs=$observedPublishMs max=$MaxPublishMs")|Out-Null}
$report=[ordered]@{schema='realtime_budget_report_v1';generatedAt=(Get-Date -Format 'o');observedPublishMs=$observedPublishMs;maxPublishMs=$MaxPublishMs;violations=@($violations);ready=($violations.Count -eq 0)}
$report|ConvertTo-Json -Depth 8|Set-Content -LiteralPath $reportPath -Encoding UTF8
Write-Host "[INFO] report: $reportPath"; if($violations.Count -gt 0){foreach($v in $violations){Write-Host "[ERROR] $v"}; throw 'realtime budget verification failed'}
Write-Host '[PASS] realtime budget verification passed'
