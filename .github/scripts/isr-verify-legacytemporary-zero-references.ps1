$ErrorActionPreference='Stop'
$repoRoot=[System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$postPath=Join-Path $repoRoot 'storage\isr_inventory\post_authority_inventory.json'
$reportPath=Join-Path $repoRoot 'evidence\legacytemporary_zero_references_report.json'
if(-not(Test-Path -LiteralPath (Split-Path $reportPath -Parent))){New-Item -ItemType Directory -Path (Split-Path $reportPath -Parent) -Force|Out-Null}
$violations=New-Object 'System.Collections.Generic.List[string]'
$legacyCount=0
if(Test-Path -LiteralPath $postPath){$j=Get-Content -LiteralPath $postPath -Raw -Encoding UTF8|ConvertFrom-Json; $legacy=@($j.entries|Where-Object{"$($_.authority_class)" -eq 'LegacyTemporary'}); $legacyCount=$legacy.Count } else { $violations.Add("Missing inventory: $postPath")|Out-Null }
if($legacyCount -gt 0){$violations.Add("LegacyTemporary references remain: count=$legacyCount")|Out-Null}
$report=[ordered]@{schema='legacytemporary_zero_references_report_v1';generatedAt=(Get-Date -Format 'o');inventoryPath=$postPath;legacyCount=$legacyCount;violations=@($violations);ready=($violations.Count -eq 0)}
$report|ConvertTo-Json -Depth 8|Set-Content -LiteralPath $reportPath -Encoding UTF8
Write-Host "[INFO] report: $reportPath"; if($violations.Count -gt 0){foreach($v in $violations){Write-Host "[ERROR] $v"}; throw 'legacytemporary zero references verification failed'}
Write-Host '[PASS] legacytemporary zero references verification passed'
