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
    $violations.Add('RuntimePublicationCoordinator ctor must initialize state to Bootstrapping')
}

if (-not [regex]::IsMatch($cppText, 'if \(boundary != RuntimeBoundary::NonRTWorld \|\| newWorld == nullptr\)\s*\{\s*convo::publishAtomic\(state_, CoordinatorState::Faulted')) {
    $violations.Add('commit must fail-closed to Faulted on invalid boundary/newWorld')
}

if (-not [regex]::IsMatch($cppText, 'convo::publishAtomic\(state_, CoordinatorState::Publishing') -or
    -not [regex]::IsMatch($cppText, 'convo::publishAtomic\(currentWorld_, newWorld') -or
    -not [regex]::IsMatch($cppText, 'convo::publishAtomic\(state_, CoordinatorState::Ready')) {
    $violations.Add('commit must implement Publishing -> currentWorld publish -> Ready sequence')
}

if (-not [regex]::IsMatch($cppText, 'if \(previousWorld != nullptr && version <= previousVersion\)\s*\{\s*convo::publishAtomic\(state_, CoordinatorState::Faulted')) {
    $violations.Add('commit must enforce monotonic generation (non-increasing version => Faulted)')
}

if (-not [regex]::IsMatch($cppText, 'if \(boundary != RuntimeBoundary::NonRTWorld \|\| oldWorld == nullptr\)\s*\{\s*convo::publishAtomic\(state_, CoordinatorState::Faulted')) {
    $violations.Add('retire must fail-closed to Faulted on invalid boundary/oldWorld')
}

$retiredWorldPushMatches = [regex]::Matches($cppText, 'retiredWorlds_\.push_back\(')
if ($retiredWorldPushMatches.Count -ne 1) {
    $violations.Add("retiredWorlds_ push authority must be single-owner (expected=1 actual=$($retiredWorldPushMatches.Count))")
}

if (-not [regex]::IsMatch($cppText, 'void RuntimePublicationCoordinator::retire\([\s\S]*?retiredWorlds_\.push_back\(oldWorld\)')) {
    $violations.Add('retiredWorlds_ push authority must be implemented inside RuntimePublicationCoordinator::retire')
}

if (-not [regex]::IsMatch($cppText, 'if \(backlog > 0\) \{\s*convo::publishAtomic\(state_, CoordinatorState::Pressure') -and
    -not [regex]::IsMatch($cppText, 'if \(slope > kPressureSlopeThreshold\) \{\s*convo::publishAtomic\(pressureNormalizedWindows_, static_cast<std::uint32_t>\(0\), std::memory_order_release\);\s*convo::publishAtomic\(state_, CoordinatorState::Pressure')) {
    $violations.Add('retire/setRetireBacklogCount must transition to Pressure on configured pressure signal (backlog or slope threshold)')
}

if (-not [regex]::IsMatch($cppText, 'if \(state == CoordinatorState::Pressure \|\| state == CoordinatorState::Publishing\) \{[\s\S]*?CoordinatorState::Ready') -and
    -not [regex]::IsMatch($cppText, 'if \(state == CoordinatorState::Pressure\) \{[\s\S]*?if \(nextWindow < kPressureNormalizeWindows\) \{\s*return;\s*\}[\s\S]*?CoordinatorState::Ready')) {
    $violations.Add('setRetireBacklogCount must restore Ready when pressure is normalized and swap is not pending')
}

if (-not [regex]::IsMatch($cppText, 'void RuntimePublicationCoordinator::markTransitionStart\(\) noexcept \{[\s\S]*?CoordinatorState::Transitioning')) {
    $violations.Add('markTransitionStart must transition state to Transitioning')
}

if (-not [regex]::IsMatch($cppText, 'void RuntimePublicationCoordinator::markTransitionStart\(\) noexcept \{\s*const auto state = convo::consumeAtomic\(state_, std::memory_order_acquire\);\s*if \(state != CoordinatorState::Ready\) \{\s*return;\s*\}')) {
    $violations.Add('markTransitionStart must reject requests when current coordinator state is not Ready')
}

if (-not [regex]::IsMatch($cppText, 'void RuntimePublicationCoordinator::markTransitionStart\(\) noexcept \{[\s\S]*?if \(getCurrent\(\) == nullptr\) \{\s*convo::publishAtomic\(state_, CoordinatorState::Faulted')) {
    $violations.Add('markTransitionStart must fail-closed to Faulted when current world is null')
}

if (-not [regex]::IsMatch($cppText, 'void RuntimePublicationCoordinator::markTransitionCommitted\(\) noexcept \{[\s\S]*?if \(!isSwapPending\(\)\) \{[\s\S]*?CoordinatorState::Ready')) {
    $violations.Add('markTransitionCommitted must transition to Ready when swap is not pending')
}

if (-not [regex]::IsMatch($cppText, 'void RuntimePublicationCoordinator::markTransitionCommitted\(\) noexcept \{\s*const auto state = convo::consumeAtomic\(state_, std::memory_order_acquire\);\s*if \(state != CoordinatorState::Transitioning\) \{\s*return;\s*\}')) {
    $violations.Add('markTransitionCommitted must reject requests when current coordinator state is not Transitioning')
}

if (-not [regex]::IsMatch($cppText, 'void RuntimePublicationCoordinator::markTransitionCommitted\(\) noexcept \{[\s\S]*?if \(!isSwapPending\(\)\) \{\s*if \(getCurrent\(\) != nullptr\) \{\s*convo::publishAtomic\(state_, CoordinatorState::Ready\, std::memory_order_release\);\s*\}\s*else \{\s*convo::publishAtomic\(state_, CoordinatorState::Faulted')) {
    $violations.Add('markTransitionCommitted must keep non-null world invariant for Ready transition')
}

if (-not [regex]::IsMatch($cppText, 'if \(state == CoordinatorState::Pressure \|\| state == CoordinatorState::Publishing\) \{\s*if \(getCurrent\(\) != nullptr\) \{\s*convo::publishAtomic\(state_, CoordinatorState::Ready\, std::memory_order_release\);\s*\}\s*else \{\s*convo::publishAtomic\(state_, CoordinatorState::Faulted') -and
    -not [regex]::IsMatch($cppText, 'if \(state == CoordinatorState::Pressure\) \{[\s\S]*?if \(getCurrent\(\) != nullptr\) \{[\s\S]*?CoordinatorState::Ready[\s\S]*?\}\s*else \{\s*convo::publishAtomic\(state_, CoordinatorState::Faulted')) {
    $violations.Add('setRetireBacklogCount must guard Ready transition with non-null current world invariant')
}

if (-not [regex]::IsMatch($cppText, 'void RuntimePublicationCoordinator::requestShutdown\(\) noexcept \{\s*convo::publishAtomic\(state_, CoordinatorState::ShuttingDown')) {
    $violations.Add('requestShutdown must transition state to ShuttingDown')
}

if (-not [regex]::IsMatch($cppText, 'void RuntimePublicationCoordinator::markShutdownComplete\(\) noexcept \{\s*const auto state = convo::consumeAtomic\(state_, std::memory_order_acquire\);\s*if \(state != CoordinatorState::ShuttingDown\) \{\s*return;\s*\}')) {
    $violations.Add('markShutdownComplete must reject requests when current coordinator state is not ShuttingDown')
}

if (-not [regex]::IsMatch($cppText, 'if \(isFullyDrained\(\)\) \{\s*convo::publishAtomic\(state_, CoordinatorState::Bootstrapping') -or
    -not [regex]::IsMatch($cppText, 'else \{\s*convo::publishAtomic\(state_, CoordinatorState::Faulted')) {
    $violations.Add('markShutdownComplete must branch Bootstrapping/Faulted by full-drain result')
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
