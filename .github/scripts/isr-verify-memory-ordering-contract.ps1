$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$schemaPath = Join-Path $repoRoot 'src\audioengine\ISRRuntimeSemanticSchema.h'
$commitPath = Join-Path $repoRoot 'src\audioengine\AudioEngine.Commit.cpp'
$threadingPath = Join-Path $repoRoot 'src\audioengine\AudioEngine.Threading.cpp'
$coordinatorPath = Join-Path $repoRoot 'src\audioengine\ISRRuntimePublicationCoordinator.cpp'
$coordinatorHeaderPath = Join-Path $repoRoot 'src\audioengine\ISRRuntimePublicationCoordinator.h'
$evidenceDir = Join-Path $repoRoot 'evidence'
$reportPath = Join-Path $evidenceDir 'memory_ordering_contract_report.json'

if (-not (Test-Path -LiteralPath $evidenceDir)) {
    New-Item -ItemType Directory -Path $evidenceDir -Force | Out-Null
}

$violations = New-Object 'System.Collections.Generic.List[string]'

foreach ($path in @($schemaPath, $commitPath, $threadingPath, $coordinatorPath, $coordinatorHeaderPath)) {
    if (-not (Test-Path -LiteralPath $path)) {
        $violations.Add("Missing required file: $path") | Out-Null
    }
}

$schemaText = if (Test-Path -LiteralPath $schemaPath) { Get-Content -LiteralPath $schemaPath -Raw -Encoding UTF8 } else { '' }
$commitText = if (Test-Path -LiteralPath $commitPath) { Get-Content -LiteralPath $commitPath -Raw -Encoding UTF8 } else { '' }
$threadingText = if (Test-Path -LiteralPath $threadingPath) { Get-Content -LiteralPath $threadingPath -Raw -Encoding UTF8 } else { '' }
$coordinatorText = if (Test-Path -LiteralPath $coordinatorPath) { Get-Content -LiteralPath $coordinatorPath -Raw -Encoding UTF8 } else { '' }
$coordinatorHeaderText = if (Test-Path -LiteralPath $coordinatorHeaderPath) { Get-Content -LiteralPath $coordinatorHeaderPath -Raw -Encoding UTF8 } else { '' }

# isMonotonic はヘッダファイルのインライン定義なので両方を検索
$combinedCoordinatorText = $coordinatorText + "`n" + $coordinatorHeaderText

if (-not $schemaText.Contains('"MemoryOrderingContractVerifier"')) {
    $violations.Add('kRequiredVerifierTable must register MemoryOrderingContractVerifier') | Out-Null
}

$commitPatterns = @(
    'consumeAtomic\(lastCommittedRuntimeGeneration_, std::memory_order_acquire\)',
    'consumeAtomic\(lastCommittedPublicationSequence_, std::memory_order_acquire\)',
    'publishAtomic\(lastCommittedRuntimeGeneration_, world.generation, std::memory_order_release\)',
    'publishAtomic\(lastCommittedPublicationSequence_, world.publication.sequenceId, std::memory_order_release\)'
)

foreach ($pattern in $commitPatterns) {
    if (-not [regex]::IsMatch($commitText, $pattern, [System.Text.RegularExpressions.RegexOptions]::Singleline)) {
        $violations.Add("Memory ordering contract pattern missing in AudioEngine.Commit.cpp: $pattern") | Out-Null
    }
}

# retirePressurePublicationThrottleActive_ は削除済（過去のリファクタリングで除去）
# retirePressureAdmissionStrict_ の publish は AudioEngine.Retire.cpp / Timer.cpp で確認済
$retireText = Get-Content -LiteralPath (Join-Path $repoRoot 'src\audioengine\AudioEngine.Retire.cpp') -Raw -Encoding UTF8 -ErrorAction SilentlyContinue
if (-not [regex]::IsMatch($retireText, 'publishAtomic\(retirePressureAdmissionStrict_, true, std::memory_order_release\)')) {
    $violations.Add('AudioEngine.Retire.cpp must publish retire admission strict gate with memory_order_release') | Out-Null
}

if (-not [regex]::IsMatch($coordinatorText,
    'const auto prev = persistentState_;')) {
    $violations.Add('ISRRuntimePublicationCoordinator.cpp must read persistentState_ before monotonic checks') | Out-Null
}

if (-not [regex]::IsMatch($coordinatorText,
    'persistentState_\s*=\s*PersistentStateBlock\{')) {
    $violations.Add('ISRRuntimePublicationCoordinator.cpp must assign persistentState_ after monotonic checks') | Out-Null
}

# isMonotonic 冁E��実裁E�E監査�E�Eeturn false への改悪を防止�E�E
# isMonotonic はヘッダのインライン定義なので combinedCoordinatorText を使用
if (-not [regex]::IsMatch($combinedCoordinatorText,
    'nextSeqId > prev\.publicationSequenceId')) {
    $violations.Add('isMonotonic(): sequenceId strict monotonic contract violated') | Out-Null
}
if (-not [regex]::IsMatch($combinedCoordinatorText,
    'nextEpoch > prev\.publicationEpoch')) {
    $violations.Add('isMonotonic(): epoch strict monotonic contract violated') | Out-Null
}
if (-not [regex]::IsMatch($combinedCoordinatorText,
    'nextGen > prev\.mappedRuntimeGeneration')) {
    $violations.Add('isMonotonic(): mappedGeneration strict monotonic contract violated') | Out-Null
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
