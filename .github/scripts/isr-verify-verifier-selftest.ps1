$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$evidenceDir = Join-Path $repoRoot 'evidence'
$reportPath = Join-Path $evidenceDir 'verifier_selftest_report.json'

if (-not (Test-Path -LiteralPath $evidenceDir)) {
    New-Item -ItemType Directory -Path $evidenceDir -Force | Out-Null
}

$violations = New-Object 'System.Collections.Generic.List[string]'

function Add-Violation {
    param([string]$Message)
    $violations.Add($Message) | Out-Null
}

function Assert-True {
    param(
        [bool]$Condition,
        [string]$Message
    )

    if (-not $Condition) {
        Add-Violation $Message
    }
}

# 1) Required verifier scripts must exist.
$requiredScripts = @(
    '.github/scripts/isr-run-tiered-verification.ps1',
    '.github/scripts/isr-verify-verifier-execution-layers.ps1',
    '.github/scripts/isr-verify-governance-registries.ps1',
    '.github/scripts/isr-verify-shadow-compare-contract.ps1',
    '.github/scripts/isr-verify-shadow-compare-cadence.ps1'
)

foreach ($scriptRelPath in $requiredScripts) {
    $scriptAbsPath = Join-Path $repoRoot $scriptRelPath
    Assert-True -Condition (Test-Path -LiteralPath $scriptAbsPath) -Message "Missing required verifier script: $scriptRelPath"
}

# 2) Synthetic negative test: duplicate verifier names must be detected by self-test logic.
function Test-DuplicateVerifierNameDetection {
    param([string[]]$Names)

    $seen = @{}
    foreach ($name in $Names) {
        if ($seen.ContainsKey($name)) {
            return $true
        }
        $seen[$name] = $true
    }

    return $false
}

$hasDuplicate = Test-DuplicateVerifierNameDetection -Names @('a', 'b', 'a')
Assert-True -Condition $hasDuplicate -Message 'Self-test failed: duplicate verifier name synthetic case was not detected.'

$hasNoDuplicate = Test-DuplicateVerifierNameDetection -Names @('a', 'b', 'c')
Assert-True -Condition (-not $hasNoDuplicate) -Message 'Self-test failed: false positive on duplicate verifier name detection.'

# 3) Defensive lint: disallow trivial always-true verifier bodies.
$scriptPaths = Get-ChildItem -LiteralPath (Join-Path $repoRoot '.github/scripts') -Filter 'isr-verify-*.ps1' -File -ErrorAction SilentlyContinue
foreach ($path in $scriptPaths) {
    $text = Get-Content -LiteralPath $path.FullName -Raw -Encoding UTF8
    if ($text -match '^\s*return\s+\$true\s*$') {
        Add-Violation "Potentially trivial verifier body detected (single-line return `$true): $($path.Name)"
    }
}

$report = [ordered]@{
    schema       = 'verifier_selftest_report_v1'
    generatedAt  = (Get-Date -Format 'o')
    required     = $requiredScripts
    scannedCount = @($scriptPaths).Count
    violations   = @($violations)
    ready        = ($violations.Count -eq 0)
}

$report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $reportPath -Encoding UTF8
Write-Host "[INFO] verifier self-test report written: $reportPath"

if ($violations.Count -gt 0) {
    foreach ($v in $violations) {
        Write-Host "[ERROR] $v"
    }
    throw "Verifier self-test failed. violations=$($violations.Count)"
}

Write-Host '[PASS] verifier self-test passed'
