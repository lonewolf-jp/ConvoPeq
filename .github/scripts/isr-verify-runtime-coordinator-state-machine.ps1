$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$headerPath = Join-Path $repoRoot 'src\audioengine\ISRRuntimePublicationCoordinator.h'
$cppPath = Join-Path $repoRoot 'src\audioengine\ISRRuntimePublicationCoordinator.cpp'
$evidenceDir = Join-Path $repoRoot 'evidence'
$reportPath = Join-Path $evidenceDir 'runtime_coordinator_state_machine_report.json'

if (-not (Test-Path -LiteralPath $evidenceDir)) {
    New-Item -ItemType Directory -Path $evidenceDir -Force | Out-Null
}

foreach ($requiredPath in @($headerPath, $cppPath)) {
    if (-not (Test-Path -LiteralPath $requiredPath)) {
        throw "Missing RuntimeCoordinator source file: $requiredPath"
    }
}

$headerText = Get-Content -LiteralPath $headerPath -Raw -Encoding UTF8
$cppText = Get-Content -LiteralPath $cppPath -Raw -Encoding UTF8

$violations = New-Object 'System.Collections.Generic.List[string]'

$requiredStates = @('Bootstrapping', 'Ready', 'Publishing', 'Transitioning', 'Pressure', 'ShuttingDown', 'Faulted')
foreach ($stateName in $requiredStates) {
    if (-not [regex]::IsMatch($headerText, "\b$([regex]::Escape($stateName))\b")) {
        $violations.Add("CoordinatorState missing required state: $stateName")
    }
}

if (-not [regex]::IsMatch($cppText, 'state_\(CoordinatorState::Bootstrapping\)')) {
    $violations.Add('RuntimeIntentCoordinator ctor must initialize state to Bootstrapping')
}

if (-not [regex]::IsMatch($cppText, 'if \(boundary != RuntimeBoundary::NonRTWorld \|\| newWorld == nullptr\)\s*\{\s*convo::publishAtomic\(state_, CoordinatorState::Faulted')) {
    $violations.Add('commit must fail-closed to Faulted on invalid boundary/newWorld')
}

if (-not [regex]::IsMatch($cppText, 'convo::publishAtomic\(state_, CoordinatorState::Publishing') -or
    -not [regex]::IsMatch($cppText, 'convo::publishAtomic\(swapPending_, true') -or
    -not [regex]::IsMatch($cppText, 'pubWorld->publication = PublicationSemantic\{') -or
    -not [regex]::IsMatch($cppText, 'convo::publishAtomic\(swapPending_, false') -or
    -not [regex]::IsMatch($cppText, 'convo::publishAtomic\(state_, CoordinatorState::Ready')) {
    $violations.Add('commit must implement Publishing -> swapPending(true) -> metadata publish -> swapPending(false) -> Ready sequence')
}

# ★ work93 refresh (dash2 §1.6.1 同期): sequence/epoch の単調性は wraparound-safe modular
#   comparison（convo::isr::isAfter / SequenceArithmetic.h）で強制するのが现行契約
#   （isAfter(a,b)==(a>b) at 非 wrap 値・seq/epoch は +1 増加のため semantics-preserving）。
#   mappedGeneration は raw 比較維持。旧条項の static_cast<uint64_t> raw > regex は
#   c8ca439b (2026-08-11) 時点形で、3b43a35d (2026-09-10, work89 Phase H) の意図的再設計が
#   未反映だった → 検査側を现行契约へ同期（production code を旧形へ戻す方向は禁止）。
if (-not [regex]::IsMatch($cppText, 'convo::isr::isAfter\(sequenceId, prevSeqId\)') -or
    -not [regex]::IsMatch($cppText, 'convo::isr::isAfter\(epoch, prevEpoch\)') -or
    -not [regex]::IsMatch($cppText, 'mappedGeneration > prevGen')) {
    $violations.Add('commit must enforce monotonic sequence via wraparound-safe isAfter(seq/epoch) + gen comparison (non-monotonic => Faulted)')
}

if (-not [regex]::IsMatch($cppText, 'if \(boundary != RuntimeBoundary::NonRTWorld \|\| oldWorld == nullptr\)\s*\{\s*convo::publishAtomic\(state_, CoordinatorState::Faulted')) {
    $violations.Add('retire must fail-closed to Faulted on invalid boundary/oldWorld')
}

# ★ work93 refresh (dash2 §1.4 同期): 现行 accounting contract —
#   retire() は retireBacklogCount_ を直接増やさない（元 setRetireBacklogCount 上書き設計の
#   commit 毎無制限増加回帰を防止。retire の実測判定は Layer 1 が pendingRetireCount +
#   pendingIntentCount を直参照）。accounting の本体は semantic event 対
#   onRetireAccepted()（fetch_add + noteRetireBacklogChanged）/ onRetireConsumed()
#   （old>0 underflow guard + fetch_sub、違反時 Faulted）。setRetireBacklogCount は
#   TEST-ONLY（production 絶対値上書き禁止）。旧条項「retire must update backlog through
#   setRetireBacklogCount(backlog+1)」は c8ca439b (08-11) 時点形で 3b43a35d (09-10) の
#   意図的再設計が未反映 → 検査側を现行契约へ同期（production code は変更しない）。
$retireFn = [regex]::Match($cppText, 'void RuntimeIntentCoordinator::retire\([^{]*\{[\s\S]*?\n\}')
if (-not $retireFn.Success) {
    $violations.Add('retire(...) definition not found for backlog contract check')
} elseif ([regex]::IsMatch($retireFn.Value, 'fetchAddAtomic\(retireBacklogCount_')) {
    $violations.Add('retire must NOT increment retireBacklogCount_ directly (dash2 §1.4: accounting via onRetireAccepted/onRetireConsumed)')
}
if (-not [regex]::IsMatch($cppText, 'void RuntimeIntentCoordinator::onRetireAccepted\(\) noexcept \{[\s\S]*?fetchAddAtomic\(retireBacklogCount_')) {
    $violations.Add('onRetireAccepted must account backlog via fetchAddAtomic(retireBacklogCount_ ...) (dash2 §1.4)')
}
if (-not [regex]::IsMatch($cppText, 'void RuntimeIntentCoordinator::onRetireConsumed\(\) noexcept \{[\s\S]*?if \(old > 0\)[\s\S]*?fetchSubAtomic\(retireBacklogCount_')) {
    $violations.Add('onRetireConsumed must guard underflow (old > 0) before fetchSubAtomic(retireBacklogCount_ ...) (dash2 §1.4)')
}
if (-not [regex]::IsMatch($cppText, 'TEST-ONLY[\s\S]{0,160}void RuntimeIntentCoordinator::setRetireBacklogCount')) {
    $violations.Add('setRetireBacklogCount must remain TEST-ONLY marked (dash2 §1.4: production absolute-value overwrite prohibited)')
}

if (-not [regex]::IsMatch($cppText, 'if \(backlog > 0\) \{\s*convo::publishAtomic\(state_, CoordinatorState::Pressure') -and
    -not [regex]::IsMatch($cppText, 'if \(slope > kPressureSlopeThreshold\) \{\s*convo::publishAtomic\(pressureNormalizedWindows_, static_cast<std::uint32_t>\(0\), std::memory_order_release\);\s*convo::publishAtomic\(state_, CoordinatorState::Pressure')) {
    $violations.Add('retire/setRetireBacklogCount must transition to Pressure on configured pressure signal (backlog or slope threshold)')
}

if (-not [regex]::IsMatch($cppText, 'if \(state == CoordinatorState::Pressure \|\| state == CoordinatorState::Publishing\) \{[\s\S]*?CoordinatorState::Ready') -and
    -not [regex]::IsMatch($cppText, 'if \(state == CoordinatorState::Pressure\) \{[\s\S]*?if \(nextWindow < kPressureNormalizeWindows\) \{\s*return;\s*\}[\s\S]*?CoordinatorState::Ready')) {
    $violations.Add('setRetireBacklogCount must restore Ready when pressure is normalized and swap is not pending')
}

if (-not [regex]::IsMatch($cppText, 'void RuntimeIntentCoordinator::markTransitionStart\(\) noexcept \{[\s\S]*?CoordinatorState::Transitioning')) {
    $violations.Add('markTransitionStart must transition state to Transitioning')
}

if (-not [regex]::IsMatch($cppText, 'void RuntimeIntentCoordinator::markTransitionStart\(\) noexcept \{\s*const auto state = convo::consumeAtomic\(state_, std::memory_order_acquire\);\s*if \(state != CoordinatorState::Ready\) \{\s*return;\s*\}')) {
    $violations.Add('markTransitionStart must reject requests when current coordinator state is not Ready')
}

if (-not [regex]::IsMatch($cppText, 'void RuntimeIntentCoordinator::markTransitionCommitted\(\) noexcept \{[\s\S]*?if \(!isSwapPending\(\)\) \{[\s\S]*?CoordinatorState::Ready')) {
    $violations.Add('markTransitionCommitted must transition to Ready when swap is not pending')
}

if (-not [regex]::IsMatch($cppText, 'void RuntimeIntentCoordinator::markTransitionCommitted\(\) noexcept \{\s*const auto state = convo::consumeAtomic\(state_, std::memory_order_acquire\);\s*if \(state != CoordinatorState::Transitioning\) \{\s*return;\s*\}')) {
    $violations.Add('markTransitionCommitted must reject requests when current coordinator state is not Transitioning')
}

if (-not [regex]::IsMatch($cppText, 'void RuntimeIntentCoordinator::requestShutdown\(\) noexcept \{') -or
    (-not [regex]::IsMatch($cppText, 'convo::publishAtomic\(state_, CoordinatorState::ShuttingDown') -and
     -not [regex]::IsMatch($cppText, 'ShutdownScheduler::requestShutdown\(\) noexcept \{\s*convo::publishAtomic\(coordinator_\.state_, CoordinatorState::ShuttingDown'))) {
    $violations.Add('requestShutdown must transition state to ShuttingDown (directly or via ShutdownScheduler)')
}

if (-not [regex]::IsMatch($cppText, 'void RuntimeIntentCoordinator::markShutdownComplete\(\) noexcept \{') -or
    (-not [regex]::IsMatch($cppText, 'ShutdownScheduler::markShutdownComplete\(\) noexcept \{\s*const auto state = convo::consumeAtomic\(coordinator_\.state_, std::memory_order_acquire\);\s*if \(state != CoordinatorState::ShuttingDown\) \{\s*return;\s*\}') -and
     -not [regex]::IsMatch($cppText, 'void RuntimeIntentCoordinator::markShutdownComplete\(\) noexcept \{\s*const auto state = convo::consumeAtomic\(state_, std::memory_order_acquire\);\s*if \(state != CoordinatorState::ShuttingDown\) \{\s*return;\s*\}'))) {
    $violations.Add('markShutdownComplete must reject requests when current coordinator state is not ShuttingDown (directly or via ShutdownScheduler)')
}

if ((-not [regex]::IsMatch($cppText, 'if \(isFullyDrained\(\)\) \{\s*convo::publishAtomic\(state_, CoordinatorState::Bootstrapping') -and
     -not [regex]::IsMatch($cppText, 'if \(isFullyDrained\(\)\) \{\s*convo::publishAtomic\(coordinator_\.state_, CoordinatorState::Bootstrapping')) -or
    ((-not [regex]::IsMatch($cppText, 'else \{\s*convo::publishAtomic\(state_, CoordinatorState::Faulted') -and
      -not [regex]::IsMatch($cppText, 'else \{\s*convo::publishAtomic\(coordinator_\.state_, CoordinatorState::Faulted')))) {
    $violations.Add('markShutdownComplete must branch Bootstrapping/Faulted by full-drain result (directly or via ShutdownScheduler)')
}

$report = [ordered]@{
    schema         = 'runtime_coordinator_state_machine_report_v1'
    generatedAt    = (Get-Date -Format 'o')
    headerPath     = $headerPath
    sourcePath     = $cppPath
    requiredStates = @($requiredStates)
    violations     = @($violations)
    ready          = ($violations.Count -eq 0)
}

$report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $reportPath -Encoding UTF8
Write-Host "[INFO] runtime coordinator state machine report written: $reportPath"

if ($violations.Count -gt 0) {
    foreach ($violation in $violations) {
        Write-Host "[ERROR] $violation"
    }
    throw "RuntimeCoordinator state machine verification failed. violations=$($violations.Count)"
}

Write-Host '[PASS] RuntimeCoordinator state machine gate verified'
