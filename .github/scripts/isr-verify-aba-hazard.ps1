$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$schemaPath = Join-Path $repoRoot 'src\audioengine\ISRRuntimeSemanticSchema.h'
$headerPath = Join-Path $repoRoot 'src\audioengine\AudioEngine.h'
$commitPath = Join-Path $repoRoot 'src\audioengine\AudioEngine.Commit.cpp'
$coordinatorPath = Join-Path $repoRoot 'src\audioengine\ISRRuntimePublicationCoordinator.cpp'
$evidenceDir = Join-Path $repoRoot 'evidence'
$reportPath = Join-Path $evidenceDir 'aba_hazard_report.json'

if (-not (Test-Path -LiteralPath $evidenceDir)) {
    New-Item -ItemType Directory -Path $evidenceDir -Force | Out-Null
}

$violations = New-Object 'System.Collections.Generic.List[string]'

foreach ($path in @($schemaPath, $headerPath, $commitPath, $coordinatorPath)) {
    if (-not (Test-Path -LiteralPath $path)) {
        $violations.Add("Missing required file: $path") | Out-Null
    }
}

$schemaText = if (Test-Path -LiteralPath $schemaPath) { Get-Content -LiteralPath $schemaPath -Raw -Encoding UTF8 } else { '' }
$headerText = if (Test-Path -LiteralPath $headerPath) { Get-Content -LiteralPath $headerPath -Raw -Encoding UTF8 } else { '' }
$commitText = if (Test-Path -LiteralPath $commitPath) { Get-Content -LiteralPath $commitPath -Raw -Encoding UTF8 } else { '' }
$coordinatorText = if (Test-Path -LiteralPath $coordinatorPath) { Get-Content -LiteralPath $coordinatorPath -Raw -Encoding UTF8 } else { '' }

if (-not $schemaText.Contains('"ABAHazardVerifier"')) {
    $violations.Add('kRequiredVerifierTable must register ABAHazardVerifier') | Out-Null
}

$headerPatterns = @(
    'RuntimeWorldIdGenerator runtimeWorldIdGenerator_',
    'worldOwner->worldId = nextWorldId;',
    'worldOwner->generation = nextGraphGeneration;',
    'worldOwner->retire.retireEpoch = nextGraphGeneration;',
    'worldOwner->publication.sequenceId = nextPublicationSequence;',
    'worldOwner->publication.mappedRuntimeGeneration = nextGraphGeneration;'
)

foreach ($pattern in $headerPatterns) {
    if (-not [regex]::IsMatch($headerText, [regex]::Escape($pattern))) {
        $violations.Add("ABA identity contract pattern missing in AudioEngine.h: $pattern") | Out-Null
    }
}

$commitPatterns = @(
    'if \(targetWorldIdU64 <= lastEnqueuedTargetWorldId\)',
    'intent->requestId = convo::fetchAddAtomic\(publicationIntentRequestIdCounter_,',
    'publishAtomic\(lastEnqueuedPublicationTargetWorldId_, targetWorldIdU64, std::memory_order_release\)'
)

foreach ($pattern in $commitPatterns) {
    if (-not [regex]::IsMatch($commitText, $pattern, [System.Text.RegularExpressions.RegexOptions]::Singleline)) {
        $violations.Add("ABA queue monotonicity pattern missing in AudioEngine.Commit.cpp: $pattern") | Out-Null
    }
}

$coordinatorPatterns = @(
    'const auto previousSequenceId = convo::consumeAtomic\(publicationSequenceId_, std::memory_order_acquire\);',
    'const auto previousMappedGeneration = convo::consumeAtomic\(mappedRuntimeGeneration_, std::memory_order_acquire\);',
    'if \(hasPrevious && sequenceId <= previousSequenceId\)',
    'if \(hasPrevious\s*&& epoch > previousEpoch\s*&& mappedGeneration < previousMappedGeneration\)',
    'publishAtomic\(publicationSequenceId_, sequenceId, std::memory_order_release\)',
    'publishAtomic\(mappedRuntimeGeneration_, mappedGeneration, std::memory_order_release\)'
)

foreach ($pattern in $coordinatorPatterns) {
    if (-not [regex]::IsMatch($coordinatorText, $pattern, [System.Text.RegularExpressions.RegexOptions]::Singleline)) {
        $violations.Add("ABA contract pattern missing in ISRRuntimePublicationCoordinator.cpp: $pattern") | Out-Null
    }
}

$report = [ordered]@{
    schema = 'aba_hazard_report_v1'
    generatedAt = (Get-Date -Format 'o')
    headerPath = $headerPath
    commitPath = $commitPath
    coordinatorPath = $coordinatorPath
    violations = @($violations)
    ready = ($violations.Count -eq 0)
}

$report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $reportPath -Encoding UTF8
Write-Host "[INFO] report: $reportPath"
if ($violations.Count -gt 0) {
    foreach ($v in $violations) { Write-Host "[ERROR] $v" }
    throw 'ABA hazard verification failed'
}

Write-Host '[PASS] ABA hazard verification passed'
