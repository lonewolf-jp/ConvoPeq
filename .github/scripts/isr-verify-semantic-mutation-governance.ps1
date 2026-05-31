$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$postPath = Join-Path $repoRoot 'storage\isr_inventory\post_authority_inventory.json'
$evidenceDir = Join-Path $repoRoot 'evidence'
$reportPath = Join-Path $evidenceDir 'semantic_mutation_governance_report.json'

if (-not (Test-Path -LiteralPath $evidenceDir)) { New-Item -ItemType Directory -Path $evidenceDir -Force | Out-Null }
if (-not (Test-Path -LiteralPath $postPath)) { throw "Missing inventory: $postPath" }

$post = Get-Content -LiteralPath $postPath -Raw -Encoding UTF8 | ConvertFrom-Json
$entries = @($post.entries)
$violations = New-Object 'System.Collections.Generic.List[string]'

$audioThreadWriters = @($entries | Where-Object { @($_.writers) -contains 'AudioThread' })
foreach ($e in $audioThreadWriters) {
    $violations.Add("AudioThread writer forbidden: state=$($e.state)") | Out-Null
}

$nonRTReaderMissing = @($entries | Where-Object { @($_.readers) -notcontains 'NonRT' })
foreach ($e in $nonRTReaderMissing) {
    $violations.Add("NonRT reader missing: state=$($e.state)") | Out-Null
}

$report = [ordered]@{ schema='semantic_mutation_governance_report_v1'; generatedAt=(Get-Date -Format 'o'); total=$entries.Count; audioThreadWriterCount=$audioThreadWriters.Count; nonRTReaderMissingCount=$nonRTReaderMissing.Count; violations=@($violations); ready=($violations.Count -eq 0) }
$report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $reportPath -Encoding UTF8
Write-Host "[INFO] report: $reportPath"
if ($violations.Count -gt 0) { foreach($v in $violations){ Write-Host "[ERROR] $v" }; throw 'semantic mutation governance verification failed' }
Write-Host '[PASS] semantic mutation governance verification passed'
