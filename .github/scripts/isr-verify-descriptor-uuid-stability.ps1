$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$postPath = Join-Path $repoRoot 'storage\isr_inventory\post_authority_inventory.json'
$evidenceDir = Join-Path $repoRoot 'evidence'
$reportPath = Join-Path $evidenceDir 'descriptor_uuid_stability_report.json'

if (-not (Test-Path -LiteralPath $evidenceDir)) { New-Item -ItemType Directory -Path $evidenceDir -Force | Out-Null }
if (-not (Test-Path -LiteralPath $postPath)) { throw "Missing inventory: $postPath" }

$post = Get-Content -LiteralPath $postPath -Raw -Encoding UTF8 | ConvertFrom-Json
$entries = @($post.entries)
$violations = New-Object 'System.Collections.Generic.List[string]'

$uuidEntries = @($entries | Where-Object { "$($_.state)" -match 'runtimeUuid|worldId' })
if ($uuidEntries.Count -lt 2) {
    $violations.Add("UUID descriptor coverage too low: expected>=2 actual=$($uuidEntries.Count)") | Out-Null
}

$duplicateStates = @($uuidEntries | Group-Object -Property state | Where-Object { $_.Count -gt 1 } | ForEach-Object { $_.Name })
foreach ($d in $duplicateStates) {
    $violations.Add("Duplicate UUID descriptor state: $d") | Out-Null
}

$report = [ordered]@{ schema='descriptor_uuid_stability_report_v1'; generatedAt=(Get-Date -Format 'o'); uuidStateCount=$uuidEntries.Count; duplicateStates=@($duplicateStates); violations=@($violations); ready=($violations.Count -eq 0) }
$report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $reportPath -Encoding UTF8
Write-Host "[INFO] report: $reportPath"
if ($violations.Count -gt 0) { foreach ($v in $violations) { Write-Host "[ERROR] $v" }; throw 'descriptor uuid stability verification failed' }
Write-Host '[PASS] descriptor uuid stability verification passed'
