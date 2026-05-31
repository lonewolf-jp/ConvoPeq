$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$schemaPath = Join-Path $repoRoot 'src\audioengine\ISRRuntimeSemanticSchema.h'
$audioHeaderPath = Join-Path $repoRoot 'src\audioengine\AudioEngine.h'
$evidenceDir = Join-Path $repoRoot 'evidence'
$reportPath = Join-Path $evidenceDir 'visibility_monotonicity_report.json'

if (-not (Test-Path -LiteralPath $evidenceDir)) {
    New-Item -ItemType Directory -Path $evidenceDir -Force | Out-Null
}

$violations = New-Object 'System.Collections.Generic.List[string]'

foreach ($path in @($schemaPath, $audioHeaderPath)) {
    if (-not (Test-Path -LiteralPath $path)) {
        $violations.Add("Missing required file: $path") | Out-Null
    }
}

$schemaText = if (Test-Path -LiteralPath $schemaPath) { Get-Content -LiteralPath $schemaPath -Raw -Encoding UTF8 } else { '' }
$audioHeaderText = if (Test-Path -LiteralPath $audioHeaderPath) { Get-Content -LiteralPath $audioHeaderPath -Raw -Encoding UTF8 } else { '' }

if (-not $schemaText.Contains('"VisibilityMonotonicityVerifier"')) {
    $violations.Add('kRequiredVerifierTable must register VisibilityMonotonicityVerifier') | Out-Null
}

$requiredPatterns = @(
    'const auto currentGeneration = world->generation;',
    'const auto previousGeneration = consumeAtomic\(observeLastSeenGeneration_\[slot\], std::memory_order_acquire\);',
    'const auto currentSequence = world->publication.sequenceId;',
    'const auto previousSequence = consumeAtomic\(observeLastSeenSequenceId_\[slot\], std::memory_order_acquire\);',
    'const bool generationBackward = \(previousGeneration != 0 && currentGeneration < previousGeneration\);',
    'const bool sequenceBackward = \(previousSequence != 0 && currentSequence < previousSequence\);',
    'fetchAddAtomic\(observeMonotonicViolationCount_, static_cast<std::uint64_t>\(1\), std::memory_order_acq_rel\);',
    'publishAtomic\(observeMonotonicRollbackRequested_, true, std::memory_order_release\);',
    'if \(currentGeneration > previousGeneration\)',
    'publishAtomic\(observeLastSeenGeneration_\[slot\], currentGeneration, std::memory_order_release\);',
    'if \(currentSequence > previousSequence\)',
    'publishAtomic\(observeLastSeenSequenceId_\[slot\], currentSequence, std::memory_order_release\);',
    'std::array<std::atomic<std::uint64_t>, 4> observeLastSeenGeneration_',
    'std::array<std::atomic<std::uint64_t>, 4> observeLastSeenSequenceId_',
    'std::atomic<std::uint64_t> observeMonotonicViolationCount_ \{ 0 \};',
    'std::atomic<bool> observeMonotonicRollbackRequested_ \{ false \};'
)

foreach ($pattern in $requiredPatterns) {
    if (-not [regex]::IsMatch($audioHeaderText, $pattern, [System.Text.RegularExpressions.RegexOptions]::Singleline)) {
        $violations.Add("Visibility monotonicity contract pattern missing: $pattern") | Out-Null
    }
}

$report = [ordered]@{
    schema          = 'visibility_monotonicity_report_v1'
    generatedAt     = (Get-Date -Format 'o')
    schemaPath      = $schemaPath
    audioHeaderPath = $audioHeaderPath
    violations      = @($violations)
    ready           = ($violations.Count -eq 0)
}

$report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $reportPath -Encoding UTF8
Write-Host "[INFO] report: $reportPath"
if ($violations.Count -gt 0) {
    foreach ($v in $violations) { Write-Host "[ERROR] $v" }
    throw 'visibility monotonicity verification failed'
}

Write-Host '[PASS] visibility monotonicity verification passed'
