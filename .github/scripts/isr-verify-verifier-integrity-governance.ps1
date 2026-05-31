$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$scriptsDir = Join-Path $repoRoot '.github\scripts'
$evidenceDir = Join-Path $repoRoot 'evidence'
$reportPath = Join-Path $evidenceDir 'verifier_integrity_governance_report.json'

if (-not (Test-Path -LiteralPath $evidenceDir)) {
    New-Item -ItemType Directory -Path $evidenceDir -Force | Out-Null
}

$violations = New-Object 'System.Collections.Generic.List[string]'
function Add-Violation([string]$m){ $violations.Add($m) | Out-Null }

$requiredScripts = @(
    'isr-run-tiered-verification.ps1',
    'isr-verify-governance-registries.ps1',
    'isr-verify-verifier-selftest.ps1'
)

foreach ($name in $requiredScripts) {
    $p = Join-Path $scriptsDir $name
    if (-not (Test-Path -LiteralPath $p)) {
        Add-Violation "Missing required verifier/governance script: $name"
    }
}

$allVerifierScripts = Get-ChildItem -LiteralPath $scriptsDir -Filter 'isr-verify-*.ps1' -File -ErrorAction SilentlyContinue
if (@($allVerifierScripts).Count -eq 0) {
    Add-Violation 'No isr-verify-*.ps1 scripts found in .github/scripts'
}

$dupes = $allVerifierScripts | Group-Object -Property Name | Where-Object { $_.Count -gt 1 }
foreach ($d in $dupes) {
    Add-Violation "Duplicate verifier script name detected: $($d.Name) x $($d.Count)"
}

# Basic sanity gate: verifier files must not be empty.
foreach ($f in $allVerifierScripts) {
    $text = Get-Content -LiteralPath $f.FullName -Raw -Encoding UTF8
    if ([string]::IsNullOrWhiteSpace($text)) {
        Add-Violation "Verifier file is empty: $($f.Name)"
    }
}

$report = [ordered]@{
    schema = 'verifier_integrity_governance_report_v1'
    generatedAt = (Get-Date -Format 'o')
    verifierCount = @($allVerifierScripts).Count
    requiredScripts = $requiredScripts
    violations = @($violations)
    ready = ($violations.Count -eq 0)
}

$report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $reportPath -Encoding UTF8
Write-Host "[INFO] report: $reportPath"

if ($violations.Count -gt 0) {
    foreach ($v in $violations) {
        Write-Host "[ERROR] $v"
    }
    throw "verifier integrity governance verification failed"
}

Write-Host '[PASS] verifier integrity governance verification passed'
