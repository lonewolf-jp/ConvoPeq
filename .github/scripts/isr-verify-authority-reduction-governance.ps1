$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$diffPath = Join-Path $repoRoot 'storage\isr_inventory\inventory_diff_report.json'
$allowlistPath = Join-Path $repoRoot '.github\isr-authority-growth-allowlist.json'
$evidenceDir = Join-Path $repoRoot 'evidence'
$reportPath = Join-Path $evidenceDir 'authority_reduction_governance_report.json'

if (-not (Test-Path -LiteralPath $evidenceDir)) { New-Item -ItemType Directory -Path $evidenceDir -Force | Out-Null }
if (-not (Test-Path -LiteralPath $diffPath)) { throw "Missing diff report: $diffPath" }
if (-not (Test-Path -LiteralPath $allowlistPath)) { throw "Missing allowlist: $allowlistPath" }

$diff = Get-Content -LiteralPath $diffPath -Raw -Encoding UTF8 | ConvertFrom-Json
$allow = Get-Content -LiteralPath $allowlistPath -Raw -Encoding UTF8 | ConvertFrom-Json

$addedCount = [int]$diff.summary.addedCount
$removedCount = [int]$diff.summary.removedCount
$netGrowth = $addedCount - $removedCount
$allowedNetGrowth = [int]$allow.allowedNetGrowth
$violations = New-Object 'System.Collections.Generic.List[string]'

if ("$($allow.schema)" -ne 'isr_authority_growth_allowlist_v1') {
    $violations.Add("Allowlist schema mismatch: $($allow.schema)") | Out-Null
}

try {
    $expiry = [datetime]::ParseExact("$($allow.expiry)", 'yyyy-MM-dd', [System.Globalization.CultureInfo]::InvariantCulture)
    if ((Get-Date).Date -gt $expiry.Date) {
        $violations.Add("Allowlist expired: $($allow.expiry)") | Out-Null
    }
}
catch {
    $violations.Add("Allowlist expiry format invalid: $($allow.expiry)") | Out-Null
}

if ($netGrowth -gt $allowedNetGrowth) {
    $violations.Add("Authority net growth exceeded allowlist: netGrowth=$netGrowth allowed=$allowedNetGrowth") | Out-Null
}

$report = [ordered]@{ schema='authority_reduction_governance_report_v1'; generatedAt=(Get-Date -Format 'o'); addedCount=$addedCount; removedCount=$removedCount; netGrowth=$netGrowth; allowedNetGrowth=$allowedNetGrowth; allowlistPath=$allowlistPath; violations=@($violations); ready=($violations.Count -eq 0) }
$report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $reportPath -Encoding UTF8
Write-Host "[INFO] report: $reportPath"
if ($violations.Count -gt 0) { foreach($v in $violations){ Write-Host "[ERROR] $v" }; throw 'authority reduction governance verification failed' }
Write-Host '[PASS] authority reduction governance verification passed'
