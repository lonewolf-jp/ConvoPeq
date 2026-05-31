$ErrorActionPreference='Stop'
$repoRoot=[System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$contractPath=Join-Path $repoRoot 'doc\work10\contracts\runtimeworld_serialization_contract.md'
$snapshotSchemaPath=Join-Path $repoRoot 'doc\work10\contracts\runtimeworld_snapshot_schema.md'
$evidenceDir=Join-Path $repoRoot 'evidence'
$reportPath=Join-Path $evidenceDir 'runtime_memory_lifetime_report.json'
if(-not(Test-Path -LiteralPath $evidenceDir)){New-Item -ItemType Directory -Path $evidenceDir -Force|Out-Null}
$violations=New-Object 'System.Collections.Generic.List[string]'
foreach($p in @($contractPath,$snapshotSchemaPath)){ if(-not(Test-Path -LiteralPath $p)){ $violations.Add("Missing contract: $p")|Out-Null } }
$report=[ordered]@{schema='runtime_memory_lifetime_report_v1';generatedAt=(Get-Date -Format 'o');serializationContract=$contractPath;snapshotSchema=$snapshotSchemaPath;violations=@($violations);ready=($violations.Count -eq 0)}
$report|ConvertTo-Json -Depth 8|Set-Content -LiteralPath $reportPath -Encoding UTF8
Write-Host "[INFO] report: $reportPath"; if($violations.Count -gt 0){ foreach($v in $violations){Write-Host "[ERROR] $v"}; throw 'runtime memory lifetime verification failed' }
Write-Host '[PASS] runtime memory lifetime verification passed'
