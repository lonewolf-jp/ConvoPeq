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

# ★ 2026-08-11: FUTURE-4 で persistentState_ キャッシュを廃止し、currentWorld_ から prev metadata を取得
# ★ work93 sync (dash2 §1.7 CW-3b): prev metadata の baseline は明示 prevWorld 引数
#   （RuntimeWorldAuthority が RuntimeStore::current から供給）へ移行 — currentWorld_ 参照自体が
#   read/write dependency 除去のため撤去された（意味は同一・依存が単一化されより強い）。
#   旧 regex の currentWorld_ cast 形は形式失効 → 検査側のみ同期。
if (-not [regex]::IsMatch($coordinatorText,
    'const auto prevSeqId = prevWorld \? prevWorld->publication\.sequenceId')) {
    $violations.Add('ISRRuntimePublicationCoordinator.cpp must read prev metadata from explicit prevWorld baseline (CW-3b) before monotonic checks') | Out-Null
}

# ★ 2026-08-11: FUTURE-4 で metadata は currentWorld_ に bake（persistentState_ 廃止）
if (-not [regex]::IsMatch($coordinatorText,
    'pubWorld->publication = PublicationSemantic\{')) {
    $violations.Add('ISRRuntimePublicationCoordinator.cpp must bake publication semantics onto currentWorld_ after monotonic checks') | Out-Null
}

# isMonotonic 冁E実裁EE監査EEeturn false への改悪を防止EE
# isMonotonic はヘッダのインライン定義なので combinedCoordinatorText を使用
# ★ 2026-08-11: isMonotonic メソッドは廃止し、commit() 内でインライン比較（FUTURE-4）
# ★ work93 sync (dash2 §1.6.1 Phase H): seq/epoch 比較は wraparound-safe modular
#   comparison（convo::isr::isAfter）へ進化（非 wrap 値で (a>b) と同値・+1 増加のため
#   semantics-preserving）。mappedGeneration は raw 比較維持。検査側のみ同期。
if (-not [regex]::IsMatch($coordinatorText,
    'convo::isr::isAfter\(sequenceId, prevSeqId\)')) {
    $violations.Add('commit(): sequenceId strict monotonic contract violated (isAfter modular comparison required)') | Out-Null
}
if (-not [regex]::IsMatch($coordinatorText,
    'convo::isr::isAfter\(epoch, prevEpoch\)')) {
    $violations.Add('commit(): epoch strict monotonic contract violated (isAfter modular comparison required)') | Out-Null
}
if (-not [regex]::IsMatch($coordinatorText,
    'mappedGeneration > prevGen')) {
    $violations.Add('commit(): mappedGeneration strict monotonic contract violated') | Out-Null
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
