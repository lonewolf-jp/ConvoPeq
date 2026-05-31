$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$postPath = Join-Path $repoRoot 'storage\isr_inventory\post_authority_inventory.json'
$evidenceDir = Join-Path $repoRoot 'evidence'
$reportPath = Join-Path $evidenceDir 'semantic_authority_contract_report.json'

if (-not (Test-Path -LiteralPath $evidenceDir)) { New-Item -ItemType Directory -Path $evidenceDir -Force | Out-Null }
if (-not (Test-Path -LiteralPath $postPath)) { throw "Missing inventory: $postPath" }

$post = Get-Content -LiteralPath $postPath -Raw -Encoding UTF8 | ConvertFrom-Json
$entries = @($post.entries)
$violations = New-Object 'System.Collections.Generic.List[string]'

$multiWriter = @($entries | Where-Object { @($_.writers).Count -ne 1 -or "$( @($_.writers)[0] )" -ne 'NonRT' })
foreach ($e in $multiWriter) {
    $violations.Add("Semantic authority writer contract violation: state=$($e.state) writers=$([string]::Join(',', @($e.writers)))") | Out-Null
}

$badPublication = @($entries | Where-Object { "$($_.publication_path)" -ne 'publish(RuntimeWorld*)' })
foreach ($e in $badPublication) {
    $violations.Add("Publication contract violation: state=$($e.state) path=$($e.publication_path)") | Out-Null
}

$badObserve = @($entries | Where-Object { "$($_.observe_path)" -ne 'RuntimeWorld' })
foreach ($e in $badObserve) {
    $violations.Add("Observe contract violation: state=$($e.state) observe=$($e.observe_path)") | Out-Null
}

$report = [ordered]@{ schema='semantic_authority_contract_report_v1'; generatedAt=(Get-Date -Format 'o'); total=$entries.Count; writerViolationCount=$multiWriter.Count; publicationViolationCount=$badPublication.Count; observeViolationCount=$badObserve.Count; violations=@($violations); ready=($violations.Count -eq 0) }
$report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $reportPath -Encoding UTF8
Write-Host "[INFO] report: $reportPath"
if ($violations.Count -gt 0) { foreach($v in $violations){ Write-Host "[ERROR] $v" }; throw 'semantic authority contract verification failed' }
Write-Host '[PASS] semantic authority contract verification passed'
