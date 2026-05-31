param([int]$MaxAuthorityCount=32)
$ErrorActionPreference='Stop'
$repoRoot=[System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$postPath=Join-Path $repoRoot 'storage\isr_inventory\post_authority_inventory.json'
$reportPath=Join-Path $repoRoot 'evidence\runtimeworld_budget_enforcement_report.json'
if(-not(Test-Path -LiteralPath (Split-Path $reportPath -Parent))){New-Item -ItemType Directory -Path (Split-Path $reportPath -Parent) -Force|Out-Null}
$violations=New-Object 'System.Collections.Generic.List[string]'
$count=0
if(Test-Path -LiteralPath $postPath){$j=Get-Content -LiteralPath $postPath -Raw -Encoding UTF8|ConvertFrom-Json; $count=@($j.entries).Count}else{$violations.Add("Missing inventory: $postPath")|Out-Null}
if($count -gt $MaxAuthorityCount){$violations.Add("RuntimeWorld authority budget exceeded: count=$count max=$MaxAuthorityCount")|Out-Null}
$report=[ordered]@{schema='runtimeworld_budget_enforcement_report_v1';generatedAt=(Get-Date -Format 'o');inventoryPath=$postPath;authorityCount=$count;maxAuthorityCount=$MaxAuthorityCount;violations=@($violations);ready=($violations.Count -eq 0)}
$report|ConvertTo-Json -Depth 8|Set-Content -LiteralPath $reportPath -Encoding UTF8
Write-Host "[INFO] report: $reportPath"; if($violations.Count -gt 0){foreach($v in $violations){Write-Host "[ERROR] $v"}; throw 'runtimeworld budget enforcement verification failed'}
Write-Host '[PASS] runtimeworld budget enforcement verification passed'
