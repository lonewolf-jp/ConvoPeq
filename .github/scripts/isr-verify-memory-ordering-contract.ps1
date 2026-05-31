$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$schemaPath = Join-Path $repoRoot 'src\audioengine\ISRRuntimeSemanticSchema.h'
$commitPath = Join-Path $repoRoot 'src\audioengine\AudioEngine.Commit.cpp'
$threadingPath = Join-Path $repoRoot 'src\audioengine\AudioEngine.Threading.cpp'
$coordinatorPath = Join-Path $repoRoot 'src\audioengine\ISRRuntimePublicationCoordinator.cpp'
$evidenceDir = Join-Path $repoRoot 'evidence'
$reportPath = Join-Path $evidenceDir 'memory_ordering_contract_report.json'

if (-not (Test-Path -LiteralPath $evidenceDir)) {
    New-Item -ItemType Directory -Path $evidenceDir -Force | Out-Null
}

$violations = New-Object 'System.Collections.Generic.List[string]'

foreach ($path in @($schemaPath, $commitPath, $threadingPath, $coordinatorPath)) {
    if (-not (Test-Path -LiteralPath $path)) {
        $violations.Add("Missing required file: $path") | Out-Null
    }
}

$schemaText = if (Test-Path -LiteralPath $schemaPath) { Get-Content -LiteralPath $schemaPath -Raw -Encoding UTF8 } else { '' }
$commitText = if (Test-Path -LiteralPath $commitPath) { Get-Content -LiteralPath $commitPath -Raw -Encoding UTF8 } else { '' }
$threadingText = if (Test-Path -LiteralPath $threadingPath) { Get-Content -LiteralPath $threadingPath -Raw -Encoding UTF8 } else { '' }
$coordinatorText = if (Test-Path -LiteralPath $coordinatorPath) { Get-Content -LiteralPath $coordinatorPath -Raw -Encoding UTF8 } else { '' }

if (-not $schemaText.Contains('"MemoryOrderingContractVerifier"')) {
    $violations.Add('kRequiredVerifierTable must register MemoryOrderingContractVerifier') | Out-Null
}

$commitPatterns = @(
    'consumeAtomic\(lastCommittedRuntimeGeneration_, std::memory_order_acquire\)',
    'consumeAtomic\(lastCommittedPublicationSequence_, std::memory_order_acquire\)',
    'publishAtomic\(lastCommittedRuntimeGeneration_, world.generation, std::memory_order_release\)',
    'publishAtomic\(lastCommittedPublicationSequence_, world.publication.sequenceId, std::memory_order_release\)',
    'compareExchangeAtomic\(tail->next,[\s\S]*?std::memory_order_release,[\s\S]*?std::memory_order_acquire\)',
    'consumeAtomic\(publicationLog.head, std::memory_order_acquire\)',
    'publishAtomic\(publicationLog.head, static_cast<PublicationIntent\*>\(nullptr\), std::memory_order_release\)'
)

foreach ($pattern in $commitPatterns) {
    if (-not [regex]::IsMatch($commitText, $pattern, [System.Text.RegularExpressions.RegexOptions]::Singleline)) {
        $violations.Add("Memory ordering contract pattern missing in AudioEngine.Commit.cpp: $pattern") | Out-Null
    }
}

if (-not [regex]::IsMatch($commitText, 'consumeAtomic\(retirePressurePublicationThrottleActive_, std::memory_order_acquire\)')) {
    $violations.Add('AudioEngine.Commit.cpp must consume retire pressure publication throttle gate with memory_order_acquire') | Out-Null
}

if (-not [regex]::IsMatch($threadingText, 'publishAtomic\(retirePressureAdmissionStrict_, severe, std::memory_order_release\)')) {
    $violations.Add('AudioEngine.Threading.cpp must publish retire pressure strict gate with memory_order_release') | Out-Null
}

if (-not [regex]::IsMatch($coordinatorText, 'consumeAtomic\(publicationSequenceId_, std::memory_order_acquire\)')) {
    $violations.Add('ISRRuntimePublicationCoordinator.cpp must acquire previous publicationSequenceId before monotonic checks') | Out-Null
}

if (-not [regex]::IsMatch($coordinatorText, 'publishAtomic\(publicationSequenceId_, sequenceId, std::memory_order_release\)')) {
    $violations.Add('ISRRuntimePublicationCoordinator.cpp must release publish publicationSequenceId after commit') | Out-Null
}

if (-not [regex]::IsMatch($coordinatorText, 'publishAtomic\(mappedRuntimeGeneration_, mappedGeneration, std::memory_order_release\)')) {
    $violations.Add('ISRRuntimePublicationCoordinator.cpp must release publish mappedRuntimeGeneration after commit') | Out-Null
}

$report = [ordered]@{
    schema = 'memory_ordering_contract_report_v1'
    generatedAt = (Get-Date -Format 'o')
    commitPath = $commitPath
    threadingPath = $threadingPath
    coordinatorPath = $coordinatorPath
    violations = @($violations)
    ready = ($violations.Count -eq 0)
}

$report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $reportPath -Encoding UTF8
Write-Host "[INFO] report: $reportPath"
if ($violations.Count -gt 0) {
    foreach ($v in $violations) { Write-Host "[ERROR] $v" }
    throw 'memory ordering contract verification failed'
}

Write-Host '[PASS] memory ordering contract verification passed'
