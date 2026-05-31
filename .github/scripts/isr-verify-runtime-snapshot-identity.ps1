$ErrorActionPreference='Stop'
$repoRoot=[System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$postPath=Join-Path $repoRoot 'storage\isr_inventory\post_authority_inventory.json'
$reportPath=Join-Path $repoRoot 'evidence\runtime_snapshot_identity_report.json'
if(-not(Test-Path -LiteralPath (Split-Path $reportPath -Parent))){New-Item -ItemType Directory -Path (Split-Path $reportPath -Parent) -Force|Out-Null}
$violations=New-Object 'System.Collections.Generic.List[string]'
$ids=New-Object 'System.Collections.Generic.HashSet[string]'
if(Test-Path -LiteralPath $postPath){$j=Get-Content -LiteralPath $postPath -Raw -Encoding UTF8|ConvertFrom-Json; foreach($e in @($j.entries)){ $id="$($e.state)|$($e.source_file)|$($e.source_line)"; if(-not $ids.Add($id)){ $violations.Add("Duplicate snapshot identity tuple: $id")|Out-Null } }} else { $violations.Add("Missing inventory: $postPath")|Out-Null }
$report=[ordered]@{schema='runtime_snapshot_identity_report_v1';generatedAt=(Get-Date -Format 'o');inventoryPath=$postPath;identityCount=$ids.Count;violations=@($violations);ready=($violations.Count -eq 0)}
$report|ConvertTo-Json -Depth 8|Set-Content -LiteralPath $reportPath -Encoding UTF8
Write-Host "[INFO] report: $reportPath"; if($violations.Count -gt 0){foreach($v in $violations){Write-Host "[ERROR] $v"}; throw 'runtime snapshot identity verification failed'}
Write-Host '[PASS] runtime snapshot identity verification passed'
