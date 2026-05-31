$ErrorActionPreference = 'Stop'
$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$headerPath = Join-Path $repoRoot 'src\audioengine\AudioEngine.h'
$contractPath = Join-Path $repoRoot 'doc\work10\contracts\runtimeworld_builder_governance.md'
$evidenceDir = Join-Path $repoRoot 'evidence'
$reportPath = Join-Path $evidenceDir 'runtimeworld_construction_report.json'
if (-not (Test-Path -LiteralPath $evidenceDir)) { New-Item -ItemType Directory -Path $evidenceDir -Force | Out-Null }
$violations = New-Object 'System.Collections.Generic.List[string]'
foreach($p in @($headerPath,$contractPath)){ if(-not(Test-Path -LiteralPath $p)){ $violations.Add("Missing required artifact: $p")|Out-Null } }
if (Test-Path -LiteralPath $headerPath) { $h = Get-Content -LiteralPath $headerPath -Raw -Encoding UTF8; if ($h -notmatch 'buildRuntimePublishWorld\s*\(') { $violations.Add('buildRuntimePublishWorld entrypoint missing')|Out-Null } }
$report = [ordered]@{ schema='runtimeworld_construction_report_v1'; generatedAt=(Get-Date -Format 'o'); headerPath=$headerPath; contractPath=$contractPath; violations=@($violations); ready=($violations.Count -eq 0) }
$report|ConvertTo-Json -Depth 8|Set-Content -LiteralPath $reportPath -Encoding UTF8
Write-Host "[INFO] report: $reportPath"; if($violations.Count -gt 0){ foreach($v in $violations){Write-Host "[ERROR] $v"}; throw 'runtimeworld construction verification failed' }
Write-Host '[PASS] runtimeworld construction verification passed'
