param(
    [int]$ExpectedPostCount = 12
)

$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$postPath = Join-Path $repoRoot 'storage\isr_inventory\post_authority_inventory.json'
$evidenceDir = Join-Path $repoRoot 'evidence'
$reportPath = Join-Path $evidenceDir 'authority_count_baseline_report.json'

if (-not (Test-Path -LiteralPath $evidenceDir)) { New-Item -ItemType Directory -Path $evidenceDir -Force | Out-Null }
if (-not (Test-Path -LiteralPath $postPath)) { throw "Missing inventory: $postPath" }

$post = Get-Content -LiteralPath $postPath -Raw -Encoding UTF8 | ConvertFrom-Json
$entries = @($post.entries)
$count = $entries.Count
$violations = New-Object 'System.Collections.Generic.List[string]'

if ($count -ne $ExpectedPostCount) {
    $violations.Add("Authority count baseline drift: expected=$ExpectedPostCount actual=$count") | Out-Null
}

$report = [ordered]@{ schema = 'authority_count_baseline_report_v1'; generatedAt = (Get-Date -Format 'o'); expected = $ExpectedPostCount; actual = $count; violations = @($violations); ready = ($violations.Count -eq 0) }
$report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $reportPath -Encoding UTF8
Write-Host "[INFO] report: $reportPath"
if ($violations.Count -gt 0) { foreach ($v in $violations) { Write-Host "[ERROR] $v" }; throw 'authority count baseline verification failed' }
Write-Host '[PASS] authority count baseline verification passed'
