$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$postPath = Join-Path $repoRoot 'storage\isr_inventory\post_authority_inventory.json'
$evidenceDir = Join-Path $repoRoot 'evidence'
$reportPath = Join-Path $evidenceDir 'authority_freeze_report.json'

if (-not (Test-Path -LiteralPath $evidenceDir)) { New-Item -ItemType Directory -Path $evidenceDir -Force | Out-Null }
if (-not (Test-Path -LiteralPath $postPath)) { throw "Missing inventory: $postPath" }

$post = Get-Content -LiteralPath $postPath -Raw -Encoding UTF8 | ConvertFrom-Json
$entries = @($post.entries)
$violations = New-Object 'System.Collections.Generic.List[string]'

$legacy = @($entries | Where-Object { "$($_.authority_class)" -eq 'LegacyTemporary' })
if ($legacy.Count -gt 0) { $violations.Add("LegacyTemporary entries remain: $($legacy.Count)") | Out-Null }

$invalidClass = @($entries | Where-Object { @('Authoritative','Derived','Diagnostic') -notcontains "$($_.authority_class)" })
foreach ($e in $invalidClass) { $violations.Add("Invalid authority_class: state=$($e.state) class=$($e.authority_class)") | Out-Null }

$report = [ordered]@{ schema='authority_freeze_report_v1'; generatedAt=(Get-Date -Format 'o'); total=$entries.Count; legacyCount=$legacy.Count; invalidClassCount=$invalidClass.Count; violations=@($violations); ready=($violations.Count -eq 0) }
$report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $reportPath -Encoding UTF8
Write-Host "[INFO] report: $reportPath"
if ($violations.Count -gt 0) { foreach ($v in $violations) { Write-Host "[ERROR] $v" }; throw 'authority freeze verification failed' }
Write-Host '[PASS] authority freeze verification passed'
