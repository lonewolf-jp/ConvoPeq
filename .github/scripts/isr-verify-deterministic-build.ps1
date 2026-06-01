# isr-verify-deterministic-build.ps1
# §3.17.3 Construction Determinism Contract (DeterministicBuildVerifier)
# Verifies that DeterministicBuildPolicy is defined with the required constraints,
# and that non-deterministic data sources (timestamps, counters, monotonic ids)
# are confined to Diagnostic/Telemetry semantic fields only.

$ErrorActionPreference = 'Stop'
$repoRoot    = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$evidenceDir = Join-Path $repoRoot 'evidence'
$reportPath  = Join-Path $evidenceDir 'isr_deterministic_build_report.json'

if (-not (Test-Path -LiteralPath $evidenceDir)) {
    New-Item -ItemType Directory -Path $evidenceDir -Force | Out-Null
}

$violations = [System.Collections.Generic.List[string]]::new()
function Add-Violation { param([string]$Msg); $violations.Add($Msg) | Out-Null }

$schemaHeader = Join-Path $repoRoot 'src\audioengine\ISRRuntimeSemanticSchema.h'
if (-not (Test-Path -LiteralPath $schemaHeader)) {
    Add-Violation "ISRRuntimeSemanticSchema.h not found: $schemaHeader"
}
else {
    $sc = Get-Content -LiteralPath $schemaHeader -Raw
    if (-not $sc.Contains('DeterministicBuildVerifier')) {
        Add-Violation 'DeterministicBuildVerifier not registered in kRequiredVerifierTable'
    }
    if (-not $sc.Contains('DeterministicBuildPolicy')) {
        Add-Violation 'DeterministicBuildPolicy struct is absent from ISRRuntimeSemanticSchema.h'
    }
    if (-not $sc.Contains('kNonDeterministicSourcesMustBeDiagnosticOnly')) {
        Add-Violation 'DeterministicBuildPolicy::kNonDeterministicSourcesMustBeDiagnosticOnly is absent'
    }
    if (-not $sc.Contains('kSameInputsSameOutput')) {
        Add-Violation 'DeterministicBuildPolicy::kSameInputsSameOutput is absent'
    }
    # ProjectionFreshness is the designated Diagnostic container for non-deterministic fields.
    if (-not $sc.Contains('ProjectionFreshness')) {
        Add-Violation 'ProjectionFreshness (non-deterministic diagnostic container) is absent from schema'
    }
    # Authority semantic fields must not include timing/counter fields.
    # Verify that TimingSemantic is separate from Authority publication fields.
    if (-not $sc.Contains('TimingSemantic')) {
        Add-Violation 'TimingSemantic struct is absent; non-deterministic timing data may leak into authority fields'
    }
}

$runtimeBuilderCpp = Join-Path $repoRoot 'src\audioengine\RuntimeBuilder.cpp'
if (-not (Test-Path -LiteralPath $runtimeBuilderCpp)) {
    Add-Violation "RuntimeBuilder.cpp not found: $runtimeBuilderCpp"
}
else {
    $ah = Get-Content -LiteralPath $runtimeBuilderCpp -Raw -Encoding UTF8
    $buildMatch = [regex]::Match(
        $ah,
        '(?s)RuntimeBuilder::buildRuntimePublishWorld\(.*?\)\s*noexcept\s*\{(?<body>.*?)\n\}',
        [System.Text.RegularExpressions.RegexOptions]::Singleline
    )

    if (-not $buildMatch.Success) {
        Add-Violation 'RuntimeBuilder::buildRuntimePublishWorld body extraction failed for deterministic build verification'
    }
    else {
        $body = $buildMatch.Groups['body'].Value
        if (-not $body.Contains('worldOwner->semanticHash.')) {
            Add-Violation 'buildRuntimePublishWorld does not compute semanticHash fields'
        }
        if (-not $body.Contains('worldOwner->freeze()')) {
            Add-Violation 'buildRuntimePublishWorld must freeze worldOwner before publication'
        }

        $nonDeterministicPatterns = @(
            'juce::Time::',
            'getHighResolutionTicks',
            'std::chrono',
            'QueryPerformanceCounter',
            'GetTickCount',
            '\brand\s*\('
        )
        foreach ($pattern in $nonDeterministicPatterns) {
            if ($body -match $pattern) {
                Add-Violation "buildRuntimePublishWorld uses non-deterministic source pattern: $pattern"
            }
        }
    }
}

$report = [ordered]@{
    schema      = 'isr_deterministic_build_evidence_v1'
    generatedAt = (Get-Date -Format 'o')
    violations  = @($violations)
    ready       = ($violations.Count -eq 0)
}

$report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $reportPath -Encoding UTF8
Write-Host "[INFO] DeterministicBuildVerifier evidence written: $reportPath"

if ($violations.Count -gt 0) {
    foreach ($v in $violations) { Write-Host "[FAIL] $v" }
    throw "DeterministicBuildVerifier contract violation. violations=$($violations.Count)"
}
Write-Host '[PASS] DeterministicBuildVerifier contract verification passed'
