$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$schemaPath = Join-Path $repoRoot 'src\audioengine\ISRRuntimeSemanticSchema.h'
$coordinatorHeaderPath = Join-Path $repoRoot 'src\audioengine\ISRRuntimePublicationCoordinator.h'
$coordinatorCppPath = Join-Path $repoRoot 'src\audioengine\ISRRuntimePublicationCoordinator.cpp'
$semanticUnitTestPath = Join-Path $repoRoot 'src\tests\ISRSemanticValidationTests.cpp'
$evidenceDir = Join-Path $repoRoot 'evidence'
$reportPath = Join-Path $evidenceDir 'publication_failure_taxonomy_report.json'

if (-not (Test-Path -LiteralPath $evidenceDir)) {
    New-Item -ItemType Directory -Path $evidenceDir -Force | Out-Null
}

$violations = New-Object 'System.Collections.Generic.List[string]'

foreach ($requiredPath in @($schemaPath, $coordinatorHeaderPath, $coordinatorCppPath, $semanticUnitTestPath)) {
    if (-not (Test-Path -LiteralPath $requiredPath)) {
        $violations.Add("Missing required file: $requiredPath") | Out-Null
    }
}

$schemaText = if (Test-Path -LiteralPath $schemaPath) { Get-Content -LiteralPath $schemaPath -Raw -Encoding UTF8 } else { '' }
$coordinatorHeaderText = if (Test-Path -LiteralPath $coordinatorHeaderPath) { Get-Content -LiteralPath $coordinatorHeaderPath -Raw -Encoding UTF8 } else { '' }
$coordinatorCppText = if (Test-Path -LiteralPath $coordinatorCppPath) { Get-Content -LiteralPath $coordinatorCppPath -Raw -Encoding UTF8 } else { '' }
$semanticUnitTestText = if (Test-Path -LiteralPath $semanticUnitTestPath) { Get-Content -LiteralPath $semanticUnitTestPath -Raw -Encoding UTF8 } else { '' }

if (-not [regex]::IsMatch($schemaText, '\{"PublicationFailureTaxonomyVerifier",\s*VerifierSeverity::Fatal\}')) {
    $violations.Add('kRequiredVerifierTable must register PublicationFailureTaxonomyVerifier with Fatal severity') | Out-Null
}

$requiredHeaderPatterns = @(
    'enum class RejectCode : uint8_t',
    'None = 0',
    'InvalidClosure',
    'InvalidPayloadTier'
)

foreach ($pattern in $requiredHeaderPatterns) {
    if (-not [regex]::IsMatch($coordinatorHeaderText, $pattern, [System.Text.RegularExpressions.RegexOptions]::Singleline)) {
        $violations.Add("Publication failure taxonomy header pattern missing: $pattern") | Out-Null
    }
}

$requiredCppPatterns = @(
    'publishAtomic\(lastRejectCode_, RejectCode::InvalidClosure, std::memory_order_release\);',
    'publishAtomic\(lastRejectCode_, RejectCode::InvalidPayloadTier, std::memory_order_release\);',
    'publishAtomic\(lastRejectCode_, RejectCode::None, std::memory_order_release\);',
    'case RejectCode::InvalidClosure:',
    'return "invalid closure graph";',
    'case RejectCode::InvalidPayloadTier:',
    'return "invalid payload tier";'
)

foreach ($pattern in $requiredCppPatterns) {
    if (-not [regex]::IsMatch($coordinatorCppText, $pattern, [System.Text.RegularExpressions.RegexOptions]::Singleline)) {
        $violations.Add("Publication failure taxonomy implementation pattern missing: $pattern") | Out-Null
    }
}

$requiredUnitPatterns = @(
    'testInvalidClosureRejected\(\)',
    'testInvalidTierRejected\(\)',
    'std::strcmp\(coordinator.lastRejectReason\(\), "invalid closure graph"\)',
    'std::strcmp\(coordinator.lastRejectReason\(\), "invalid payload tier"\)'
)

foreach ($pattern in $requiredUnitPatterns) {
    if (-not [regex]::IsMatch($semanticUnitTestText, $pattern, [System.Text.RegularExpressions.RegexOptions]::Singleline)) {
        $violations.Add("Publication failure taxonomy unit-test pattern missing: $pattern") | Out-Null
    }
}

$report = [ordered]@{
    schema               = 'publication_failure_taxonomy_report_v1'
    generatedAt          = (Get-Date -Format 'o')
    schemaPath           = $schemaPath
    coordinatorHeaderPath = $coordinatorHeaderPath
    coordinatorCppPath   = $coordinatorCppPath
    semanticUnitTestPath = $semanticUnitTestPath
    violations           = @($violations)
    ready                = ($violations.Count -eq 0)
}

$report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $reportPath -Encoding UTF8
Write-Host "[INFO] report: $reportPath"

if ($violations.Count -gt 0) {
    foreach ($v in $violations) { Write-Host "[ERROR] $v" }
    throw 'publication failure taxonomy verification failed'
}

Write-Host '[PASS] publication failure taxonomy verification passed'
