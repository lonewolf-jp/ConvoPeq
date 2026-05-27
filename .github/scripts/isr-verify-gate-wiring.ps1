$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$workflowPath = Join-Path $repoRoot ".github\workflows\isr-verification.yml"
$tierRunnerPath = Join-Path $repoRoot ".github\scripts\isr-run-tiered-verification.ps1"
$validatorTieringPath = Join-Path $repoRoot '.github\scripts\isr-verify-validator-tiering.ps1'
$triggerCleanupCompletionPath = Join-Path $repoRoot '.github\scripts\isr-verify-trigger-cleanup-completion.ps1'

foreach ($path in @($workflowPath, $tierRunnerPath, $validatorTieringPath, $triggerCleanupCompletionPath)) {
    if (-not (Test-Path $path)) {
        throw "Missing required file: $path"
    }
}

$workflowText = Get-Content -LiteralPath $workflowPath -Raw -Encoding UTF8
$tierRunnerText = Get-Content -LiteralPath $tierRunnerPath -Raw -Encoding UTF8
$validatorTieringText = Get-Content -LiteralPath $validatorTieringPath -Raw -Encoding UTF8
$triggerCleanupCompletionText = Get-Content -LiteralPath $triggerCleanupCompletionPath -Raw -Encoding UTF8

$requiredGateScripts = @(
    '.github/scripts/isr-verify-v1-immutability.ps1',
    '.github/scripts/isr-verify-v2-seal.ps1',
    '.github/scripts/isr-verify-v3-runtime-graph-immutability.ps1',
    '.github/scripts/isr-verify-v4-dsp-handle-policy.ps1',
    '.github/scripts/isr-verify-v5-retire-authority-lane.ps1',
    '.github/scripts/isr-verify-v6-domain-f-ordering.ps1',
    '.github/scripts/isr-verify-v7-rt-nonrt-retire-bridge.ps1',
    '.github/scripts/isr-verify-v8-shared-split-readiness.ps1',
    '.github/scripts/isr-verify-phase4-generation-drift.ps1',
    '.github/scripts/isr-verify-v3.ps1',
    '.github/scripts/isr-verify-v4.ps1',
    '.github/scripts/isr-verify-v5.ps1',
    '.github/scripts/isr-verify-v6.ps1',
    '.github/scripts/isr-verify-v7.ps1',
    '.github/scripts/isr-verify-v8.ps1',
    '.github/scripts/isr-verify-v9.ps1',
    '.github/scripts/isr-verify-v10.ps1',
    '.github/scripts/isr-verify-v10-ownership-cycle.ps1',
    '.github/scripts/isr-verify-r11-r25-closed-coverage.ps1',
    '.github/scripts/isr-verify-trigger-policy.ps1',
    '.github/scripts/isr-verify-trigger-symbol-usage.ps1',
    '.github/scripts/isr-verify-observe-shim-usage.ps1',
    '.github/scripts/isr-verify-trigger-ast.ps1',
    '.github/scripts/isr-trigger-audit.ps1',
    '.github/scripts/isr-prune-cleanup-deferred.ps1',
    '.github/scripts/isr-rebuild-admission-8_1-metrics.ps1',
    '.github/scripts/isr-verify-enforcement-adoption.ps1',
    '.github/scripts/isr-verify-enforcement-source-purity.ps1',
    '.github/scripts/isr-verify-trigger-cleanup-readiness.ps1',
    '.github/scripts/isr-verify-cleanup-deferred.ps1',
    '.github/scripts/isr-verify-flag-dependency-graph.ps1',
    '.github/scripts/isr-verify-rollback-matrix.ps1',
    '.github/scripts/isr-verify-metric-governance.ps1',
    '.github/scripts/isr-verify-8_1-close-policy.ps1',
    '.github/scripts/isr-verify-workflow-dispatch-input-policy.ps1',
    '.github/scripts/isr-verify-8_1-workflow-input-contract.ps1',
    '.github/scripts/isr-verify-8_1-workflow-input-coherence.ps1',
    '.github/scripts/isr-verify-canary-baseline-normalization.ps1',
    '.github/scripts/isr-verify-policy-top-level-governance.ps1',
    '.github/scripts/isr-verify-rtmutable-boundary.ps1',
    '.github/scripts/isr-verify-facade-bypass.ps1',
    '.github/scripts/isr-verify-latency-alignment.ps1',
    '.github/scripts/isr-verify-crossfade-observable-state.ps1',
    '.github/scripts/isr-verify-ownership-migration.ps1',
    '.github/scripts/isr-verify-validator-tiering.ps1',
    '.github/scripts/isr-verify-trigger-cleanup-completion.ps1',
    '.github/scripts/isr-verify-bridge-plan-completeness.ps1',
    '.github/scripts/isr-verify-backlog-specfixed-residual.ps1',
    '.github/scripts/isr-verify-clang-tidy-readiness.ps1',
    '.github/scripts/isr-verify-clang-tidy-audit.ps1'
)

foreach ($relativeScript in $requiredGateScripts) {
    $absoluteScript = Join-Path $repoRoot $relativeScript
    if (-not (Test-Path $absoluteScript)) {
        throw "Missing required gate script file: $relativeScript"
    }

    $workflowRefPattern = [regex]::Escape("& ./$relativeScript")
    $tierRefPattern = [regex]::Escape("'$relativeScript'")

    $wiredInWorkflow = [regex]::IsMatch($workflowText, $workflowRefPattern)
    $wiredInTierRunner = [regex]::IsMatch($tierRunnerText, $tierRefPattern)

    if (-not $wiredInWorkflow -and -not $wiredInTierRunner) {
        throw "Workflow/tier wiring missing gate invocation: $relativeScript"
    }
}

$workflowNeedsClangTidyAuditInput = [regex]::IsMatch($workflowText, 'requireClangTidyAudit\s*:')
if (-not $workflowNeedsClangTidyAuditInput) {
    throw 'Workflow dispatch input missing: requireClangTidyAudit'
}

$workflowNeedsAutoCapture81LogInput = [regex]::IsMatch($workflowText, 'autoCapture81Log\s*:')
if (-not $workflowNeedsAutoCapture81LogInput) {
    throw 'Workflow dispatch input missing: autoCapture81Log'
}

$workflowNeedsCollect81CloseEvidenceInput = [regex]::IsMatch($workflowText, 'collect81CloseEvidence\s*:')
if (-not $workflowNeedsCollect81CloseEvidenceInput) {
    throw 'Workflow dispatch input missing: collect81CloseEvidence'
}

$workflowNeedsCollect81WindowSecInput = [regex]::IsMatch($workflowText, 'collect81WindowSec\s*:')
if (-not $workflowNeedsCollect81WindowSecInput) {
    throw 'Workflow dispatch input missing: collect81WindowSec'
}

$workflowNeedsCollect81AutoCaptureTimeoutSecInput = [regex]::IsMatch($workflowText, 'collect81AutoCaptureTimeoutSec\s*:')
if (-not $workflowNeedsCollect81AutoCaptureTimeoutSecInput) {
    throw 'Workflow dispatch input missing: collect81AutoCaptureTimeoutSec'
}

$workflowNeedsCollect81SignalProbeInput = [regex]::IsMatch($workflowText, 'collect81SignalProbe\s*:')
if (-not $workflowNeedsCollect81SignalProbeInput) {
    throw 'Workflow dispatch input missing: collect81SignalProbe'
}

$workflowNeedsCollect81ProbeExitMsInput = [regex]::IsMatch($workflowText, 'collect81ProbeExitMs\s*:')
if (-not $workflowNeedsCollect81ProbeExitMsInput) {
    throw 'Workflow dispatch input missing: collect81ProbeExitMs'
}

$workflowNeedsAutoPruneCleanupDeferredInput = [regex]::IsMatch($workflowText, 'autoPruneCleanupDeferred\s*:')
if (-not $workflowNeedsAutoPruneCleanupDeferredInput) {
    throw 'Workflow dispatch input missing: autoPruneCleanupDeferred'
}

$workflowNeedsAutoPruneCleanupDeferredForward =
$workflowText.Contains('inputs.autoPruneCleanupDeferred') -and
$workflowText.Contains("'-AutoPruneCleanupDeferred'")
if (-not $workflowNeedsAutoPruneCleanupDeferredForward) {
    throw 'Workflow missing autoPruneCleanupDeferred forwarding to tier runner args'
}

$workflowNeedsEnforce81CloseDecisionInput = [regex]::IsMatch($workflowText, 'enforce81CloseDecision\s*:')
if (-not $workflowNeedsEnforce81CloseDecisionInput) {
    throw 'Workflow dispatch input missing: enforce81CloseDecision'
}

$workflowNeedsEnforce81CloseDecisionRetryMaxInput = [regex]::IsMatch($workflowText, 'enforce81CloseDecisionRetryMax\s*:')
if (-not $workflowNeedsEnforce81CloseDecisionRetryMaxInput) {
    throw 'Workflow dispatch input missing: enforce81CloseDecisionRetryMax'
}

$workflowNeedsEnforce81CloseDecisionRetryMaxForward =
$workflowText.Contains('inputs.enforce81CloseDecisionRetryMax') -and
$workflowText.Contains("'-Enforce81CloseDecisionRetryMax'")
if (-not $workflowNeedsEnforce81CloseDecisionRetryMaxForward) {
    throw 'Workflow missing enforce81CloseDecisionRetryMax forwarding to tier runner args'
}

$workflowNeeds81PolicyLoadWiring =
$workflowText.Contains('isr-8_1-close-policy.json') -and
$workflowText.Contains('isr_8_1_close_policy_v1') -and
$workflowText.Contains('Resolve-WorkflowPositiveInt') -and
$workflowText.Contains('Assert-WorkflowRange') -and
$workflowText.Contains('Assert-WorkflowPolicyExpiryGuard') -and
$workflowText.Contains('Assert-WorkflowInputContractAgainstPolicy') -and
$workflowText.Contains('Get-WorkflowInputBlock') -and
$workflowText.Contains('Get-WorkflowInputProperty') -and
$workflowText.Contains('Get-WorkflowDispatchInputNames') -and
$workflowText.Contains('Resolve-ValidatorTieringScheduleContract') -and
$workflowText.Contains('.github\isr-validator-tiering-policy.json') -and
$workflowText.Contains('isr_validator_tiering_policy_v1') -and
$workflowText.Contains('.github\isr-workflow-dispatch-input-policy.json') -and
$workflowText.Contains('isr_workflow_dispatch_input_policy_v1') -and
$workflowText.Contains('Assert-WorkflowDispatchInputPolicyAgainstPolicy') -and
$workflowText.Contains('Get-WorkflowInputOptions') -and
$workflowText.Contains('Workflow dispatch input policy violation: type mismatch') -and
$workflowText.Contains('Workflow dispatch input policy violation: required mismatch') -and
$workflowText.Contains('Workflow dispatch input policy violation: default mismatch') -and
$workflowText.Contains('Workflow dispatch input policy violation: description mismatch') -and
$workflowText.Contains('Workflow dispatch input policy boolean input has invalid default') -and
$workflowText.Contains('Workflow dispatch input policy has empty default') -and
$workflowText.Contains('Workflow dispatch input policy missing descriptionMustContain') -and
$workflowText.Contains('Workflow dispatch input policy choice input has duplicate option') -and
$workflowText.Contains('Workflow dispatch input policy choice input default is not in options') -and
$workflowText.Contains('Workflow dispatch input policy violation: workflow has duplicate option') -and
$workflowText.Contains('Workflow dispatch input policy missing required field: forwardingContract.switches') -and
$workflowText.Contains('Workflow dispatch input policy forwardingContract.switches requires non-empty switches') -and
$workflowText.Contains('Workflow dispatch input policy forwardingContract has duplicate inputName:') -and
$workflowText.Contains('Workflow dispatch input policy forwardingContract has duplicate runnerSwitch:') -and
$workflowText.Contains('Workflow dispatch input policy forwardingContract requires boolean input type:') -and
$workflowText.Contains('Workflow dispatch input policy forwardingContract references unknown input:') -and
$workflowText.Contains('Workflow dispatch input policy violation: missing tier runner switch forwarding:') -and
$workflowText.Contains('Workflow dispatch input policy has invalid expiry format:') -and
$workflowText.Contains('Workflow dispatch input policy expired: expiry=') -and
$workflowText.Contains('Workflow dispatch input policy violation: uncontracted non-8.1 workflow input detected') -and
$workflowText.Contains('Assert-WorkflowDispatchInputPolicyAgainstPolicy -Policy $workflowDispatchInputPolicy -WorkflowPath') -and
$workflowText.Contains('Unknown workflow schedule cron:') -and
$workflowText.Contains('Get-WorkflowInputContractEntry') -and
$workflowText.Contains('Resolve-WorkflowPolicyIntDefault') -and
$workflowText.Contains('workflowInputContract') -and
$workflowText.Contains('descriptionMustContain') -and
$workflowText.Contains('Workflow input contract violation: type mismatch') -and
$workflowText.Contains('Workflow input contract violation: required mismatch') -and
$workflowText.Contains('Workflow input contract violation: default mismatch') -and
$workflowText.Contains('Workflow input contract violation: description mismatch') -and
$workflowText.Contains('Workflow input contract violation: uncontracted 8.1 workflow input detected') -and
$workflowText.Contains('workflowInputContract references unknown collector field') -and
$workflowText.Contains('Assert-WorkflowInputContractAgainstPolicy -Policy $closePolicy -WorkflowPath') -and
$workflowText.Contains('expiryGuardDaysByTier') -and
$workflowText.Contains('expiry guard breached for tier=') -and
$workflowText.Contains('ParseExact') -and
$workflowText.Contains('collect81CloseEvidence is not allowed for verificationTier=') -and
$workflowText.Contains('enforce81CloseDecision is not allowed for verificationTier=') -and
$workflowText.Contains("-InputName 'collect81WindowSec'") -and
$workflowText.Contains("-InputName 'collect81AutoCaptureTimeoutSec'") -and
$workflowText.Contains("-InputName 'collect81ProbeExitMs'") -and
$workflowText.Contains("-InputName 'enforce81CloseDecisionRetryMax'") -and
$workflowText.Contains('Resolve-WorkflowPolicyIntDefault -Policy $closePolicy -InputName ''collect81WindowSec''') -and
$workflowText.Contains('Resolve-WorkflowPolicyIntDefault -Policy $closePolicy -InputName ''collect81AutoCaptureTimeoutSec''') -and
$workflowText.Contains('Resolve-WorkflowPolicyIntDefault -Policy $closePolicy -InputName ''collect81ProbeExitMs''') -and
$workflowText.Contains('Resolve-WorkflowPolicyIntDefault -Policy $closePolicy -InputName ''enforce81CloseDecisionRetryMax''')
if (-not $workflowNeeds81PolicyLoadWiring) {
    throw 'Workflow missing policy-driven fail-closed parsing/validation wiring for 8.1 inputs'
}

$closePolicyPath = Join-Path $repoRoot '.github\isr-8_1-close-policy.json'
if (-not (Test-Path $closePolicyPath)) {
    throw "Missing 8.1 close policy: $closePolicyPath"
}

$closePolicy = Get-Content -LiteralPath $closePolicyPath -Raw -Encoding UTF8 | ConvertFrom-Json
if ("$($closePolicy.schema)" -ne 'isr_8_1_close_policy_v1') {
    throw "Unexpected 8.1 close policy schema: $($closePolicy.schema)"
}

foreach ($field in @('owner', 'issue', 'rationale', 'expiry')) {
    if ($null -eq $closePolicy.$field -or [string]::IsNullOrWhiteSpace("$($closePolicy.$field)")) {
        throw "8.1 close policy missing required field: $field"
    }
}

if ($null -eq $closePolicy.collector) {
    throw '8.1 close policy missing collector section'
}

foreach ($field in @('minWindowSec', 'maxWindowSec', 'minAutoCaptureTimeoutSec', 'maxAutoCaptureTimeoutSec', 'minProbeExitMs', 'maxProbeExitMs', 'minRetryMax', 'maxRetryMax', 'allowedCollectTiers', 'allowedEnforceTiers')) {
    if ($null -eq $closePolicy.collector.$field) {
        throw "8.1 close policy collector missing required field: $field"
    }
}

if ($null -eq $closePolicy.workflowInputContract) {
    throw '8.1 close policy missing workflowInputContract'
}

if ([string]::IsNullOrWhiteSpace("$($closePolicy.workflowInputContract.descriptionMustContain)")) {
    throw '8.1 close policy workflowInputContract missing required field: descriptionMustContain'
}

if ($null -eq $closePolicy.workflowInputContract.inputs -or @($closePolicy.workflowInputContract.inputs).Count -eq 0) {
    throw '8.1 close policy workflowInputContract requires non-empty inputs'
}

if ($null -eq $closePolicy.expiryGuardDaysByTier) {
    throw '8.1 close policy missing expiryGuardDaysByTier'
}

foreach ($field in @('standard', 'exhaustive')) {
    if ($null -eq $closePolicy.expiryGuardDaysByTier.$field) {
        throw "8.1 close policy expiryGuardDaysByTier missing required field: $field"
    }
}

$symbolAllowlistPath = Join-Path $repoRoot '.github\isr-trigger-symbol-allowlist.json'
if (-not (Test-Path $symbolAllowlistPath)) {
    throw "Missing trigger symbol allowlist: $symbolAllowlistPath"
}

$triggerPolicyPath = Join-Path $repoRoot '.github\isr-trigger-policy.json'
if (-not (Test-Path $triggerPolicyPath)) {
    throw "Missing trigger policy: $triggerPolicyPath"
}

$triggerPolicy = Get-Content -LiteralPath $triggerPolicyPath -Raw -Encoding UTF8 | ConvertFrom-Json
foreach ($field in @('owner', 'issue', 'rationale', 'expiry')) {
    if ($null -eq $triggerPolicy.$field -or [string]::IsNullOrWhiteSpace("$($triggerPolicy.$field)")) {
        throw "Trigger policy missing required field: $field"
    }
}

$observeShimAllowlistPath = Join-Path $repoRoot '.github\isr-observe-shim-allowlist.json'
if (-not (Test-Path $observeShimAllowlistPath)) {
    throw "Missing observe shim allowlist: $observeShimAllowlistPath"
}

$observeShimAllowlist = Get-Content -LiteralPath $observeShimAllowlistPath -Raw -Encoding UTF8 | ConvertFrom-Json
foreach ($field in @('owner', 'issue', 'rationale', 'expiry')) {
    if ($null -eq $observeShimAllowlist.$field -or [string]::IsNullOrWhiteSpace("$($observeShimAllowlist.$field)")) {
        throw "Observe shim allowlist missing required field: $field"
    }
}

$symbolAllowlist = Get-Content -LiteralPath $symbolAllowlistPath -Raw -Encoding UTF8 | ConvertFrom-Json
foreach ($field in @('owner', 'issue', 'rationale', 'expiry')) {
    if ($null -eq $symbolAllowlist.$field -or [string]::IsNullOrWhiteSpace("$($symbolAllowlist.$field)")) {
        throw "Trigger symbol allowlist missing required field: $field"
    }
}

$cleanupDeferredPath = Join-Path $repoRoot '.github\isr-cleanup-deferred.json'
if (-not (Test-Path $cleanupDeferredPath)) {
    throw "Missing cleanup deferred registry: $cleanupDeferredPath"
}

$cleanupDeferred = Get-Content -LiteralPath $cleanupDeferredPath -Raw -Encoding UTF8 | ConvertFrom-Json
foreach ($field in @('owner', 'issue', 'rationale', 'expiry')) {
    if ($null -eq $cleanupDeferred.$field -or [string]::IsNullOrWhiteSpace("$($cleanupDeferred.$field)")) {
        throw "Cleanup deferred registry missing required field: $field"
    }
}

$cleanupPrunePolicyPath = Join-Path $repoRoot '.github\isr-cleanup-prune-policy.json'
if (-not (Test-Path $cleanupPrunePolicyPath)) {
    throw "Missing cleanup prune policy: $cleanupPrunePolicyPath"
}

$cleanupPrunePolicy = Get-Content -LiteralPath $cleanupPrunePolicyPath -Raw -Encoding UTF8 | ConvertFrom-Json
foreach ($field in @('owner', 'issue', 'rationale', 'expiry')) {
    if ($null -eq $cleanupPrunePolicy.$field -or [string]::IsNullOrWhiteSpace("$($cleanupPrunePolicy.$field)")) {
        throw "Cleanup prune policy missing required field: $field"
    }
}

if ($null -eq $cleanupPrunePolicy.readyRule) {
    throw 'Cleanup prune policy missing readyRule'
}

foreach ($field in @('comparator', 'requirePolicyEvaluationNotExpired', 'requireFiniteValues', 'issue')) {
    if ($null -eq $cleanupPrunePolicy.readyRule.$field -or [string]::IsNullOrWhiteSpace("$($cleanupPrunePolicy.readyRule.$field)")) {
        throw "Cleanup prune policy readyRule missing required field: $field"
    }
}

$rollbackMatrixPath = Join-Path $repoRoot '.github\isr-rollback-compatibility-matrix.json'
if (-not (Test-Path $rollbackMatrixPath)) {
    throw "Missing rollback compatibility matrix: $rollbackMatrixPath"
}

$validatorTieringPolicyPath = Join-Path $repoRoot '.github\isr-validator-tiering-policy.json'
if (-not (Test-Path $validatorTieringPolicyPath)) {
    throw "Missing validator tiering policy: $validatorTieringPolicyPath"
}

$workflowDispatchInputPolicyPath = Join-Path $repoRoot '.github\isr-workflow-dispatch-input-policy.json'
if (-not (Test-Path $workflowDispatchInputPolicyPath)) {
    throw "Missing workflow dispatch input policy: $workflowDispatchInputPolicyPath"
}

$validatorTieringPolicy = Get-Content -LiteralPath $validatorTieringPolicyPath -Raw -Encoding UTF8 | ConvertFrom-Json
foreach ($field in @('owner', 'issue', 'rationale', 'expiry')) {
    if ($null -eq $validatorTieringPolicy.$field -or [string]::IsNullOrWhiteSpace("$($validatorTieringPolicy.$field)")) {
        throw "Validator tiering policy missing required field: $field"
    }
}

$workflowDispatchInputPolicy = Get-Content -LiteralPath $workflowDispatchInputPolicyPath -Raw -Encoding UTF8 | ConvertFrom-Json
foreach ($field in @('schema', 'owner', 'issue', 'rationale', 'expiry', 'inputContract')) {
    if ($null -eq $workflowDispatchInputPolicy.$field -or [string]::IsNullOrWhiteSpace("$($workflowDispatchInputPolicy.$field)")) {
        throw "Workflow dispatch input policy missing required field: $field"
    }
}

if ("$($workflowDispatchInputPolicy.schema)" -ne 'isr_workflow_dispatch_input_policy_v1') {
    throw "Unexpected workflow dispatch input policy schema: $($workflowDispatchInputPolicy.schema)"
}

if ($null -eq $workflowDispatchInputPolicy.inputContract.inputs -or @($workflowDispatchInputPolicy.inputContract.inputs).Count -eq 0) {
    throw 'Workflow dispatch input policy inputContract requires non-empty inputs'
}

$rollbackMatrix = Get-Content -LiteralPath $rollbackMatrixPath -Raw -Encoding UTF8 | ConvertFrom-Json
foreach ($field in @('owner', 'issue', 'rationale', 'expiry')) {
    if ($null -eq $rollbackMatrix.$field -or [string]::IsNullOrWhiteSpace("$($rollbackMatrix.$field)")) {
        throw "Rollback compatibility matrix missing required field: $field"
    }
}

$enforcementSourcePolicyPath = Join-Path $repoRoot '.github\isr-enforcement-source-policy.json'
if (-not (Test-Path $enforcementSourcePolicyPath)) {
    throw "Missing enforcement source policy: $enforcementSourcePolicyPath"
}

$enforcementSourcePolicy = Get-Content -LiteralPath $enforcementSourcePolicyPath -Raw -Encoding UTF8 | ConvertFrom-Json
foreach ($field in @('owner', 'issue', 'rationale', 'expiry')) {
    if ($null -eq $enforcementSourcePolicy.$field -or [string]::IsNullOrWhiteSpace("$($enforcementSourcePolicy.$field)")) {
        throw "Enforcement source policy missing required field: $field"
    }
}

$enforcementSourcePurityScriptPath = Join-Path $repoRoot '.github\scripts\isr-verify-enforcement-source-purity.ps1'
if (-not (Test-Path $enforcementSourcePurityScriptPath)) {
    throw "Missing enforcement source purity script: $enforcementSourcePurityScriptPath"
}

$enforcementSourcePurityScriptText = Get-Content -LiteralPath $enforcementSourcePurityScriptPath -Raw -Encoding UTF8
if (-not $enforcementSourcePurityScriptText.Contains('retireFacadeRuntimeExecutionMetricSource')) {
    throw 'Enforcement source purity gate missing retireFacadeRuntimeExecutionMetricSource coverage'
}

$metricGovernancePath = Join-Path $repoRoot '.github\isr-metric-governance.json'
if (-not (Test-Path $metricGovernancePath)) {
    throw "Missing metric governance policy: $metricGovernancePath"
}

$metricGovernance = Get-Content -LiteralPath $metricGovernancePath -Raw -Encoding UTF8 | ConvertFrom-Json
foreach ($field in @('owner', 'issue', 'rationale', 'expiry')) {
    if ($null -eq $metricGovernance.$field -or [string]::IsNullOrWhiteSpace("$($metricGovernance.$field)")) {
        throw "Metric governance policy missing required field: $field"
    }
}

if ($null -eq $metricGovernance.normalizationPolicy) {
    throw 'Metric governance policy missing normalizationPolicy'
}

foreach ($field in @('enabled', 'baselineWindowMinutes', 'cpuThermalOsNormalization', 'bucketBy', 'strictModeRequireAllMetrics', 'issue')) {
    if ($null -eq $metricGovernance.normalizationPolicy.$field -or [string]::IsNullOrWhiteSpace("$($metricGovernance.normalizationPolicy.$field)")) {
        throw "Metric governance normalizationPolicy missing required field: $field"
    }
}

$tierRunnerNeedsClangTidyAuditSwitch = [regex]::IsMatch($tierRunnerText, 'RequireClangTidyAudit')
if (-not $tierRunnerNeedsClangTidyAuditSwitch) {
    throw 'Tier runner missing switch wiring: RequireClangTidyAudit'
}

$tierRunnerNeedsAutoCapture81LogSwitch = [regex]::IsMatch($tierRunnerText, 'AutoCapture81Log')
if (-not $tierRunnerNeedsAutoCapture81LogSwitch) {
    throw 'Tier runner missing switch wiring: AutoCapture81Log'
}

$tierRunnerNeedsCollect81CloseEvidenceSwitch = [regex]::IsMatch($tierRunnerText, 'Collect81CloseEvidence')
if (-not $tierRunnerNeedsCollect81CloseEvidenceSwitch) {
    throw 'Tier runner missing switch wiring: Collect81CloseEvidence'
}

$tierRunnerNeedsCollect81WindowSecSwitch = [regex]::IsMatch($tierRunnerText, 'Collect81WindowSec')
if (-not $tierRunnerNeedsCollect81WindowSecSwitch) {
    throw 'Tier runner missing switch wiring: Collect81WindowSec'
}

$tierRunnerNeedsCollect81AutoCaptureTimeoutSecSwitch = [regex]::IsMatch($tierRunnerText, 'Collect81AutoCaptureTimeoutSec')
if (-not $tierRunnerNeedsCollect81AutoCaptureTimeoutSecSwitch) {
    throw 'Tier runner missing switch wiring: Collect81AutoCaptureTimeoutSec'
}

$tierRunnerNeedsCollect81SignalProbeSwitch = [regex]::IsMatch($tierRunnerText, 'Collect81SignalProbe')
if (-not $tierRunnerNeedsCollect81SignalProbeSwitch) {
    throw 'Tier runner missing switch wiring: Collect81SignalProbe'
}

$tierRunnerNeedsCollect81ProbeExitMsSwitch = [regex]::IsMatch($tierRunnerText, 'Collect81ProbeExitMs')
if (-not $tierRunnerNeedsCollect81ProbeExitMsSwitch) {
    throw 'Tier runner missing switch wiring: Collect81ProbeExitMs'
}

$tierRunnerNeedsAutoPruneCleanupDeferredSwitch = [regex]::IsMatch($tierRunnerText, 'AutoPruneCleanupDeferred')
if (-not $tierRunnerNeedsAutoPruneCleanupDeferredSwitch) {
    throw 'Tier runner missing switch wiring: AutoPruneCleanupDeferred'
}

$tierRunnerNeedsEnforce81CloseDecisionSwitch = [regex]::IsMatch($tierRunnerText, 'Enforce81CloseDecision')
if (-not $tierRunnerNeedsEnforce81CloseDecisionSwitch) {
    throw 'Tier runner missing switch wiring: Enforce81CloseDecision'
}

$tierRunnerNeedsEnforce81CloseDecisionRetryMaxSwitch = [regex]::IsMatch($tierRunnerText, 'Enforce81CloseDecisionRetryMax')
if (-not $tierRunnerNeedsEnforce81CloseDecisionRetryMaxSwitch) {
    throw 'Tier runner missing switch wiring: Enforce81CloseDecisionRetryMax'
}

$tierRunnerNeedsClangTidyAuditTierForward = $tierRunnerText.Contains(".github/scripts/isr-verify-clang-tidy-audit.ps1") -and $tierRunnerText.Contains("-Tier `$Tier")
if (-not $tierRunnerNeedsClangTidyAuditTierForward) {
    throw 'Tier runner missing tier forwarding for isr-verify-clang-tidy-audit.ps1'
}

$tierRunnerNeeds81ClosePolicyTierForward =
$tierRunnerText.Contains(".github/scripts/isr-verify-8_1-close-policy.ps1") -and
$tierRunnerText.Contains('& $scriptPath -Tier $Tier')
if (-not $tierRunnerNeeds81ClosePolicyTierForward) {
    throw 'Tier runner missing tier forwarding for isr-verify-8_1-close-policy.ps1'
}

$enforcementForwardAnchor = "elseif (`$scriptPath -eq '.github/scripts/isr-verify-enforcement-adoption.ps1')"
$enforcementForwardCall = "& `$scriptPath -Tier `$Tier"
$tierRunnerNeedsEnforcementTierForward = $tierRunnerText.Contains($enforcementForwardAnchor) -and $tierRunnerText.Contains($enforcementForwardCall)
if (-not $tierRunnerNeedsEnforcementTierForward) {
    throw 'Tier runner missing tier forwarding for isr-verify-enforcement-adoption.ps1'
}

$enforcementSourceForwardAnchor = "elseif (`$scriptPath -eq '.github/scripts/isr-verify-enforcement-source-purity.ps1')"
$enforcementSourceForwardCall = '& $scriptPath -RequireAstEvidence'
$tierRunnerNeedsEnforcementSourceAstForward = $tierRunnerText.Contains($enforcementSourceForwardAnchor) -and $tierRunnerText.Contains($enforcementSourceForwardCall)
if (-not $tierRunnerNeedsEnforcementSourceAstForward) {
    throw 'Tier runner missing RequireAstEvidence forwarding for isr-verify-enforcement-source-purity.ps1'
}

$triggerCleanupCompletionForwardAnchor = "elseif (`$scriptPath -eq '.github/scripts/isr-verify-trigger-cleanup-completion.ps1')"
$triggerCleanupCompletionForwardCall = '& $scriptPath -RequireAstEvidence'
$tierRunnerNeedsTriggerCleanupCompletionAstForward = $tierRunnerText.Contains($triggerCleanupCompletionForwardAnchor) -and $tierRunnerText.Contains($triggerCleanupCompletionForwardCall)
if (-not $tierRunnerNeedsTriggerCleanupCompletionAstForward) {
    throw 'Tier runner missing RequireAstEvidence forwarding for isr-verify-trigger-cleanup-completion.ps1'
}

$backlogResidualForwardAnchor = "elseif (`$scriptPath -eq '.github/scripts/isr-verify-backlog-specfixed-residual.ps1')"
$backlogResidualForwardCall = '& $scriptPath -EnforceNoSpecFixed'
$tierRunnerNeedsBacklogResidualEnforceForward = $tierRunnerText.Contains($backlogResidualForwardAnchor) -and $tierRunnerText.Contains($backlogResidualForwardCall)
if (-not $tierRunnerNeedsBacklogResidualEnforceForward) {
    throw 'Tier runner missing EnforceNoSpecFixed forwarding for isr-verify-backlog-specfixed-residual.ps1'
}

$rebuildAdmissionAnchor = "elseif (`$scriptPath -eq '.github/scripts/isr-rebuild-admission-8_1-metrics.ps1')"
$rebuildAdmissionForwardCall = "& `$scriptPath -TryAutoCaptureOnMissingLog"
$tierRunnerNeeds81AutoCaptureForward = $tierRunnerText.Contains($rebuildAdmissionAnchor) -and $tierRunnerText.Contains($rebuildAdmissionForwardCall)
if (-not $tierRunnerNeeds81AutoCaptureForward) {
    throw 'Tier runner missing auto-capture forwarding for isr-rebuild-admission-8_1-metrics.ps1'
}

$collectorScriptPath = Join-Path $repoRoot '.github\scripts\isr-collect-rebuild-admission-8_1-close-evidence.ps1'
if (-not (Test-Path $collectorScriptPath)) {
    throw "Missing 8.1 close evidence collector script: $collectorScriptPath"
}

$collectorScriptText = Get-Content -LiteralPath $collectorScriptPath -Raw -Encoding UTF8
if (-not $collectorScriptText.Contains('Resolve-OperationalDecision')) {
    throw 'Collector missing operational decision resolver function wiring'
}

if (-not $collectorScriptText.Contains('operationalDecision')) {
    throw 'Collector missing operationalDecision report field wiring'
}

if (-not $collectorScriptText.Contains("$sourceName = 'probeDelta'")) {
    throw 'Collector missing probeDelta-first operational decision policy wiring'
}

if (-not $collectorScriptText.Contains("$sourceName = 'baseline'")) {
    throw 'Collector missing baseline fallback operational decision policy wiring'
}

if (-not $collectorScriptText.Contains('decisionPolicyVersion')) {
    throw 'Collector missing operational decision policy version wiring'
}

if (-not $collectorScriptText.Contains('sourceCandidates')) {
    throw 'Collector missing operational decision source candidates wiring'
}

$tierRunnerNeeds81CollectorForward = $tierRunnerText.Contains("& `$collectorScriptPath -WindowSec `$Collect81WindowSec -AutoCaptureTimeoutSec `$Collect81AutoCaptureTimeoutSec")
if (-not $tierRunnerNeeds81CollectorForward) {
    throw 'Tier runner missing forwarding for isr-collect-rebuild-admission-8_1-close-evidence.ps1'
}

$tierRunnerNeeds81CollectorProbeForward = $tierRunnerText.Contains('-ProbeOnInsufficientSignals -ProbeExitMs $Collect81ProbeExitMs')
if (-not $tierRunnerNeeds81CollectorProbeForward) {
    throw 'Tier runner missing probe forwarding for isr-collect-rebuild-admission-8_1-close-evidence.ps1'
}

$tierRunnerNeedsCleanupPruneForward =
$tierRunnerText.Contains(".github/scripts/isr-prune-cleanup-deferred.ps1") -and
$tierRunnerText.Contains('if ($AutoPruneCleanupDeferred)') -and
$tierRunnerText.Contains('& $scriptPath -Apply')
if (-not $tierRunnerNeedsCleanupPruneForward) {
    throw 'Tier runner missing cleanup deferred auto-prune forwarding'
}

$tierRunnerNeedsTriggerAuditAstForward =
$tierRunnerText.Contains("if (`$scriptPath -eq '.github/scripts/isr-trigger-audit.ps1')") -and
$tierRunnerText.Contains('-RequireAstEvidence')
if (-not $tierRunnerNeedsTriggerAuditAstForward) {
    throw 'Tier runner missing RequireAstEvidence forwarding for trigger audit'
}

$cleanupPruneScriptPath = Join-Path $repoRoot '.github\scripts\isr-prune-cleanup-deferred.ps1'
if (-not (Test-Path $cleanupPruneScriptPath)) {
    throw "Missing cleanup deferred prune script: $cleanupPruneScriptPath"
}

$cleanupPruneScriptText = Get-Content -LiteralPath $cleanupPruneScriptPath -Raw -Encoding UTF8
$cleanupPruneNeedsContractChecks =
$cleanupPruneScriptText.Contains('cleanup_deferred_prune_report_v1') -and
$cleanupPruneScriptText.Contains('trigger_audit_report_v1') -and
$cleanupPruneScriptText.Contains('cleanup_deferred_registry_v1') -and
$cleanupPruneScriptText.Contains('cleanup_prune_policy_v1') -and
$cleanupPruneScriptText.Contains('isr-cleanup-prune-policy.json') -and
$cleanupPruneScriptText.Contains('readyRule') -and
$cleanupPruneScriptText.Contains('[switch]$Apply')
if (-not $cleanupPruneNeedsContractChecks) {
    throw 'Cleanup deferred prune script missing contract checks'
}

$cleanupDeferredVerifyScriptPath = Join-Path $repoRoot '.github\scripts\isr-verify-cleanup-deferred.ps1'
if (-not (Test-Path $cleanupDeferredVerifyScriptPath)) {
    throw "Missing cleanup deferred verifier script: $cleanupDeferredVerifyScriptPath"
}

$cleanupDeferredVerifyScriptText = Get-Content -LiteralPath $cleanupDeferredVerifyScriptPath -Raw -Encoding UTF8
$cleanupDeferredNeedsPruneGovernance =
$cleanupDeferredVerifyScriptText.Contains('cleanup_deferred_prune_report_v1') -and
$cleanupDeferredVerifyScriptText.Contains('Missing cleanup deferred prune report') -and
$cleanupDeferredVerifyScriptText.Contains('pruneSummary') -and
$cleanupDeferredVerifyScriptText.Contains('prune report contains violations') -and
$cleanupDeferredVerifyScriptText.Contains('remainingCount mismatch') -and
$cleanupDeferredVerifyScriptText.Contains('apply=false requires prunedCount=0')
if (-not $cleanupDeferredNeedsPruneGovernance) {
    throw 'Cleanup deferred verifier missing prune report governance checks'
}

$tierRunnerNeeds81OperationalDecisionEnforce =
$tierRunnerText.Contains('8.1 close decision enforce failed') -and
$tierRunnerText.Contains('operationalDecision') -and
$tierRunnerText.Contains('[PASS] 8.1 close operational decision enforced')
if (-not $tierRunnerNeeds81OperationalDecisionEnforce) {
    throw 'Tier runner missing 8.1 close operational decision enforce wiring'
}

$tierRunnerNeeds81TransientRetryWiring =
$tierRunnerText.Contains('max81CloseDecisionAttempts') -and
$tierRunnerText.Contains('attempt81CloseDecision') -and
$tierRunnerText.Contains('timeoutForcedDispatchSeen') -and
$tierRunnerText.Contains('Retrying collector once.')
if (-not $tierRunnerNeeds81TransientRetryWiring) {
    throw 'Tier runner missing 8.1 transient retry wiring for timeoutForcedDispatchSeen'
}

$tierRunnerNeeds81RetryMaxConfigGuard = $tierRunnerText.Contains('Invalid configuration: Enforce81CloseDecisionRetryMax must be >= 1.')
if (-not $tierRunnerNeeds81RetryMaxConfigGuard) {
    throw 'Tier runner missing Enforce81CloseDecisionRetryMax configuration guard'
}

$tierRunnerNeeds81PolicyWiring =
$tierRunnerText.Contains('.github\isr-8_1-close-policy.json') -and
$tierRunnerText.Contains('isr_8_1_close_policy_v1') -and
$tierRunnerText.Contains('workflowInputContract') -and
$tierRunnerText.Contains('Resolve-WorkflowInputContractIntDefault') -and
$tierRunnerText.Contains('Get-WorkflowInputContractEntry') -and
$tierRunnerText.Contains('Resolve-WorkflowInputContractIntDefault -Policy $closePolicy -InputName ''collect81WindowSec''') -and
$tierRunnerText.Contains('Resolve-WorkflowInputContractIntDefault -Policy $closePolicy -InputName ''collect81AutoCaptureTimeoutSec''') -and
$tierRunnerText.Contains('Resolve-WorkflowInputContractIntDefault -Policy $closePolicy -InputName ''collect81ProbeExitMs''') -and
$tierRunnerText.Contains('Resolve-WorkflowInputContractIntDefault -Policy $closePolicy -InputName ''enforce81CloseDecisionRetryMax''') -and
$tierRunnerText.Contains('allowedCollectTiers') -and
$tierRunnerText.Contains('allowedEnforceTiers') -and
$tierRunnerText.Contains('Collect81CloseEvidence is not allowed for Tier=') -and
$tierRunnerText.Contains('Enforce81CloseDecision is not allowed for Tier=') -and
$tierRunnerText.Contains('Collect81WindowSec must be between') -and
$tierRunnerText.Contains('Collect81AutoCaptureTimeoutSec must be between') -and
$tierRunnerText.Contains('Collect81ProbeExitMs must be between') -and
$tierRunnerText.Contains('Enforce81CloseDecisionRetryMax must be between')
if (-not $tierRunnerNeeds81PolicyWiring) {
    throw 'Tier runner missing policy-driven 8.1 close parameter validation wiring'
}

$tierRunnerNeeds81OperationalDecisionPolicyContract =
$tierRunnerText.Contains('unexpected decisionPolicyVersion=') -and
$tierRunnerText.Contains('sourceCandidates contract mismatch') -and
$tierRunnerText.Contains('8.1-close-ops-v3')
if (-not $tierRunnerNeeds81OperationalDecisionPolicyContract) {
    throw 'Tier runner missing 8.1 close operational decision policy contract enforcement'
}

$tierRunnerNeeds81EnforceConfigGuard = $tierRunnerText.Contains('Invalid configuration: Enforce81CloseDecision requires Collect81CloseEvidence.')
if (-not $tierRunnerNeeds81EnforceConfigGuard) {
    throw 'Tier runner missing Enforce81CloseDecision configuration guard'
}

$tierRunnerNeedsAutoPruneTierGuard = $tierRunnerText.Contains('Invalid configuration: AutoPruneCleanupDeferred requires Tier=standard or exhaustive.')
if (-not $tierRunnerNeedsAutoPruneTierGuard) {
    throw 'Tier runner missing AutoPruneCleanupDeferred tier guard'
}

$workflowNeeds81EnforceConfigGuard = $workflowText.Contains('Invalid workflow inputs: enforce81CloseDecision requires collect81CloseEvidence=true')
if (-not $workflowNeeds81EnforceConfigGuard) {
    throw 'Workflow missing enforce81CloseDecision configuration guard'
}

$workflowNeedsAutoPruneTierGuard = $workflowText.Contains('Invalid workflow inputs: autoPruneCleanupDeferred requires verificationTier=standard or exhaustive')
if (-not $workflowNeedsAutoPruneTierGuard) {
    throw 'Workflow missing autoPruneCleanupDeferred tier guard'
}

$validatorTieringNeedsSourceRootScan = $validatorTieringText.Contains("$sourceRoot = Join-Path `$repoRoot 'src'")
if (-not $validatorTieringNeedsSourceRootScan) {
    throw 'Validator tiering gate missing runtime source root scan wiring'
}

$validatorTieringNeedsPolicyWiring = $validatorTieringText.Contains('.github\isr-validator-tiering-policy.json') -and $validatorTieringText.Contains('$policyPath')
if (-not $validatorTieringNeedsPolicyWiring) {
    throw 'Validator tiering gate missing policy wiring'
}

$validatorTieringNeedsWorkflowScheduleContract =
$validatorTieringText.Contains('.github\workflows\isr-verification.yml') -and
$validatorTieringText.Contains('workflowSchedule object') -and
$validatorTieringText.Contains('Workflow schedule contract mismatch') -and
$validatorTieringText.Contains('validator_tiering_report_v3')
if (-not $validatorTieringNeedsWorkflowScheduleContract) {
    throw 'Validator tiering gate missing workflow schedule contract wiring'
}

$policyTopLevelScriptPath = Join-Path $repoRoot '.github\scripts\isr-verify-policy-top-level-governance.ps1'
if (-not (Test-Path $policyTopLevelScriptPath)) {
    throw "Missing policy top-level governance script: $policyTopLevelScriptPath"
}

$policyTopLevelScriptText = Get-Content -LiteralPath $policyTopLevelScriptPath -Raw -Encoding UTF8
$policyTopLevelNeedsSchemaGovernance =
$policyTopLevelScriptText.Contains('missing required top-level field: schema') -and
$policyTopLevelScriptText.Contains('Duplicate policy schema detected:') -and
$policyTopLevelScriptText.Contains('Duplicate policy issue detected:') -and
$policyTopLevelScriptText.Contains('policy_top_level_governance_report_v2')
if (-not $policyTopLevelNeedsSchemaGovernance) {
    throw 'Policy top-level governance gate missing schema/duplicate-schema/duplicate-issue fail-closed checks'
}

$workflowDispatchInputPolicyScriptPath = Join-Path $repoRoot '.github\scripts\isr-verify-workflow-dispatch-input-policy.ps1'
if (-not (Test-Path $workflowDispatchInputPolicyScriptPath)) {
    throw "Missing workflow dispatch input policy gate script: $workflowDispatchInputPolicyScriptPath"
}

$workflowDispatchInputPolicyScriptText = Get-Content -LiteralPath $workflowDispatchInputPolicyScriptPath -Raw -Encoding UTF8
$workflowDispatchInputPolicyNeedsContracts =
$workflowDispatchInputPolicyScriptText.Contains('workflow_dispatch_input_policy_report.json') -and
$workflowDispatchInputPolicyScriptText.Contains('workflow_dispatch_input_policy_report_v1') -and
$workflowDispatchInputPolicyScriptText.Contains('isr-workflow-dispatch-input-policy.json') -and
$workflowDispatchInputPolicyScriptText.Contains('isr_workflow_dispatch_input_policy_v1') -and
$workflowDispatchInputPolicyScriptText.Contains('Workflow dispatch input mismatch: uncontracted non-8.1 workflow input detected') -and
$workflowDispatchInputPolicyScriptText.Contains('Workflow dispatch input policy missing required field: forwardingContract.switches') -and
$workflowDispatchInputPolicyScriptText.Contains('Workflow dispatch input policy forwardingContract.switches requires non-empty switches') -and
$workflowDispatchInputPolicyScriptText.Contains('Workflow dispatch input policy forwardingContract has duplicate inputName:') -and
$workflowDispatchInputPolicyScriptText.Contains('Workflow dispatch input policy forwardingContract has duplicate runnerSwitch:') -and
$workflowDispatchInputPolicyScriptText.Contains('Workflow dispatch input policy forwardingContract requires boolean input type:') -and
$workflowDispatchInputPolicyScriptText.Contains('Workflow dispatch forwarding mismatch: missing tier runner switch forwarding:') -and
$workflowDispatchInputPolicyScriptText.Contains('boolean input has invalid default') -and
$workflowDispatchInputPolicyScriptText.Contains('choice input has duplicate option') -and
$workflowDispatchInputPolicyScriptText.Contains('choice input default is not in options') -and
$workflowDispatchInputPolicyScriptText.Contains('Get-WorkflowInputBlock') -and
$workflowDispatchInputPolicyScriptText.Contains('Get-WorkflowInputProperty') -and
$workflowDispatchInputPolicyScriptText.Contains('Get-WorkflowDispatchInputNames') -and
$workflowDispatchInputPolicyScriptText.Contains('Get-WorkflowInputOptions')
if (-not $workflowDispatchInputPolicyNeedsContracts) {
    throw 'Workflow dispatch input policy gate missing core contract checks'
}

$triggerSymbolUsageScriptPath = Join-Path $repoRoot '.github\scripts\isr-verify-trigger-symbol-usage.ps1'
if (-not (Test-Path $triggerSymbolUsageScriptPath)) {
    throw "Missing trigger symbol usage script: $triggerSymbolUsageScriptPath"
}

$triggerAuditScriptPath = Join-Path $repoRoot '.github\scripts\isr-trigger-audit.ps1'
if (-not (Test-Path $triggerAuditScriptPath)) {
    throw "Missing trigger audit script: $triggerAuditScriptPath"
}

$triggerAuditScriptText = Get-Content -LiteralPath $triggerAuditScriptPath -Raw -Encoding UTF8
$triggerAuditNeedsSourceContractChecks =
$triggerAuditScriptText.Contains('Trigger audit source contract violated') -and
$triggerAuditScriptText.Contains('Trigger audit source contract violations detected') -and
$triggerAuditScriptText.Contains('isr-verify-trigger-symbol-usage.ps1') -and
$triggerAuditScriptText.Contains('isr-verify-observe-shim-usage.ps1') -and
$triggerAuditScriptText.Contains('isr-verify-trigger-ast.ps1') -and
$triggerAuditScriptText.Contains('[switch]$RequireAstEvidence') -and
$triggerAuditScriptText.Contains('-RequireAst') -and
$triggerAuditScriptText.Contains('Test-SourcePrefix')
if (-not $triggerAuditNeedsSourceContractChecks) {
    throw 'Trigger audit gate missing source contract hardening checks'
}

$triggerAuditNeedsEvidenceFirstContracts =
$triggerAuditScriptText.Contains('Missing required evidence report') -and
$triggerAuditScriptText.Contains('Unexpected trigger symbol usage report schema') -and
$triggerAuditScriptText.Contains('Unexpected observe shim usage report schema') -and
$triggerAuditScriptText.Contains('Unexpected trigger AST report schema') -and
$triggerAuditScriptText.Contains('Trigger AST report must be available=true for trigger audit') -and
$triggerAuditScriptText.Contains('Trigger AST report must be commandOk=true for trigger audit') -and
$triggerAuditScriptText.Contains('Trigger AST report must be required=true when trigger audit RequireAstEvidence is specified') -and
$triggerAuditScriptText.Contains('Trigger AST report effective source must be astOnly when RequireAstEvidence is specified') -and
$triggerAuditScriptText.Contains('astEvidenceRequired') -and
$triggerAuditScriptText.Contains('[bool]$RequireAstEvidence') -and
$triggerAuditScriptText.Contains('Missing required evidence generator script')
if (-not $triggerAuditNeedsEvidenceFirstContracts) {
    throw 'Trigger audit gate missing evidence-first contract checks'
}

if ($triggerAuditScriptText.Contains('Falling back to raw regex metric')) {
    throw 'Trigger audit gate must not allow raw regex fallback warning path'
}

if ($triggerAuditScriptText.Contains('function Get-MatchCount')) {
    throw 'Trigger audit gate must not rely on local source grep counting helper'
}

$triggerSymbolUsageScriptText = Get-Content -LiteralPath $triggerSymbolUsageScriptPath -Raw -Encoding UTF8
$triggerSymbolUsageNeedsRuleQualityChecks =
$triggerSymbolUsageScriptText.Contains('Duplicate allowlist rule detected') -and
$triggerSymbolUsageScriptText.Contains('Invalid allowlist pathRegex') -and
$triggerSymbolUsageScriptText.Contains('[regex]::new')
if (-not $triggerSymbolUsageNeedsRuleQualityChecks) {
    throw 'Trigger symbol usage gate missing allowlist duplicate/regex validation wiring'
}

$triggerAstScriptPath = Join-Path $repoRoot '.github\scripts\isr-verify-trigger-ast.ps1'
if (-not (Test-Path $triggerAstScriptPath)) {
    throw "Missing trigger AST script: $triggerAstScriptPath"
}

$triggerAstScriptText = Get-Content -LiteralPath $triggerAstScriptPath -Raw -Encoding UTF8
$triggerAstNeedsRequiredModeContracts =
$triggerAstScriptText.Contains('fadingOutDspWriteEffectiveSource') -and
$triggerAstScriptText.Contains("'astOnly'") -and
$triggerAstScriptText.Contains("'astOrFallbackMax'") -and
$triggerAstScriptText.Contains('AST required mode rejected fallback-dominant result')
if (-not $triggerAstNeedsRequiredModeContracts) {
    throw 'Trigger AST gate missing required-mode fail-closed contracts'
}

$phase4GenerationDriftScriptPath = Join-Path $repoRoot '.github\scripts\isr-verify-phase4-generation-drift.ps1'
if (-not (Test-Path $phase4GenerationDriftScriptPath)) {
    throw "Missing phase4 generation drift script: $phase4GenerationDriftScriptPath"
}

$phase4GenerationDriftScriptText = Get-Content -LiteralPath $phase4GenerationDriftScriptPath -Raw -Encoding UTF8
$phase4GenerationDriftNeedsEvidenceContracts =
$phase4GenerationDriftScriptText.Contains('phase4_generation_drift_report.json') -and
$phase4GenerationDriftScriptText.Contains('phase4_generation_drift_report_v1') -and
$phase4GenerationDriftScriptText.Contains('requiredFiles') -and
$phase4GenerationDriftScriptText.Contains('violations')
if (-not $phase4GenerationDriftNeedsEvidenceContracts) {
    throw 'Phase4 generation drift gate missing evidence report contracts'
}

$validatorTieringNeedsSlaFreshnessChecks =
$validatorTieringText.Contains('hb_violation_report.json') -and
$validatorTieringText.Contains('payload_tier_report.json') -and
$validatorTieringText.Contains('Resolve-ArtifactTimestampUtc') -and
$validatorTieringText.Contains('SLA breach:')
if (-not $validatorTieringNeedsSlaFreshnessChecks) {
    throw 'Validator tiering gate missing SLA freshness checks wiring'
}

$metricGovernanceScriptPath = Join-Path $repoRoot '.github\scripts\isr-verify-metric-governance.ps1'
if (-not (Test-Path $metricGovernanceScriptPath)) {
    throw "Missing metric governance script: $metricGovernanceScriptPath"
}

$rollbackMatrixScriptPath = Join-Path $repoRoot '.github\scripts\isr-verify-rollback-matrix.ps1'
if (-not (Test-Path $rollbackMatrixScriptPath)) {
    throw "Missing rollback matrix script: $rollbackMatrixScriptPath"
}

$rollbackMatrixScriptText = Get-Content -LiteralPath $rollbackMatrixScriptPath -Raw -Encoding UTF8
$rollbackMatrixNeedsRollbackReadyInvariant =
$rollbackMatrixScriptText.Contains('rollbackReady') -and
$rollbackMatrixScriptText.Contains('rollbackFlags.global') -and
$rollbackMatrixScriptText.Contains('rollbackFlags.retirePathOnly') -and
$rollbackMatrixScriptText.Contains('rollbackReady invariant mismatch')
if (-not $rollbackMatrixNeedsRollbackReadyInvariant) {
    throw 'Rollback matrix gate missing rollbackReady invariant checks'
}

$metricGovernanceScriptText = Get-Content -LiteralPath $metricGovernanceScriptPath -Raw -Encoding UTF8
$metricGovernanceNeedsNormalizationChecks =
$metricGovernanceScriptText.Contains('normalizationPolicy') -and
$metricGovernanceScriptText.Contains('baselineWindowNormalized') -and
$metricGovernanceScriptText.Contains('metric_governance_report_v2')
if (-not $metricGovernanceNeedsNormalizationChecks) {
    throw 'Metric governance gate missing baseline normalization contract checks'
}

$rtMutableBoundaryScriptPath = Join-Path $repoRoot '.github\scripts\isr-verify-rtmutable-boundary.ps1'
if (-not (Test-Path $rtMutableBoundaryScriptPath)) {
    throw "Missing RT mutable boundary script: $rtMutableBoundaryScriptPath"
}

$rtMutableBoundaryScriptText = Get-Content -LiteralPath $rtMutableBoundaryScriptPath -Raw -Encoding UTF8
$rtMutableBoundaryNeedsRTAuxPointerBan =
$rtMutableBoundaryScriptText.Contains('forbiddenRTAuxFieldDeclarationPatterns') -and
$rtMutableBoundaryScriptText.Contains('RTAuxMutable contains forbidden field declaration pattern') -and
$rtMutableBoundaryScriptText.Contains('NonOwningPtr')
if (-not $rtMutableBoundaryNeedsRTAuxPointerBan) {
    throw 'RT mutable boundary gate missing RTAuxMutable pointer/ownership field checks'
}

$canaryNormalizationScriptPath = Join-Path $repoRoot '.github\scripts\isr-verify-canary-baseline-normalization.ps1'
if (-not (Test-Path $canaryNormalizationScriptPath)) {
    throw "Missing canary baseline normalization script: $canaryNormalizationScriptPath"
}

$canaryNormalizationScriptText = Get-Content -LiteralPath $canaryNormalizationScriptPath -Raw -Encoding UTF8
$canaryNormalizationNeedsContractChecks =
$canaryNormalizationScriptText.Contains('canary_baseline_normalization_report_v1') -and
$canaryNormalizationScriptText.Contains('ISR_REQUIRE_RUNTIME_EVIDENCE') -and
$canaryNormalizationScriptText.Contains('baselineWindowNormalized') -and
$canaryNormalizationScriptText.Contains('strictModeRequireAllMetrics')
if (-not $canaryNormalizationNeedsContractChecks) {
    throw 'Canary baseline normalization gate missing contract checks'
}

$facadeBypassScriptPath = Join-Path $repoRoot '.github\scripts\isr-verify-facade-bypass.ps1'
if (-not (Test-Path $facadeBypassScriptPath)) {
    throw "Missing facade bypass script: $facadeBypassScriptPath"
}

$facadeBypassScriptText = Get-Content -LiteralPath $facadeBypassScriptPath -Raw -Encoding UTF8
$facadeBypassNeedsTriggerAuditConsistency =
$facadeBypassScriptText.Contains('trigger_audit_report.json') -and
$facadeBypassScriptText.Contains('trigger_audit_report_v1') -and
$facadeBypassScriptText.Contains('retireFacadeRuntimeExecutionCount') -and
$facadeBypassScriptText.Contains('retireFacadeRawDependencyCount')
if (-not $facadeBypassNeedsTriggerAuditConsistency) {
    throw 'Facade bypass gate missing trigger audit metric consistency checks'
}

$bridgePlanCompletenessScriptPath = Join-Path $repoRoot '.github\scripts\isr-verify-bridge-plan-completeness.ps1'
if (-not (Test-Path $bridgePlanCompletenessScriptPath)) {
    throw "Missing bridge plan completeness script: $bridgePlanCompletenessScriptPath"
}

$bridgePlanCompletenessScriptText = Get-Content -LiteralPath $bridgePlanCompletenessScriptPath -Raw -Encoding UTF8
$bridgePlanCompletenessNeedsContracts =
$bridgePlanCompletenessScriptText.Contains('bridge_plan_completeness_report.json') -and
$bridgePlanCompletenessScriptText.Contains('bridge_plan_completeness_report_v1') -and
$bridgePlanCompletenessScriptText.Contains('Phase 6: cleanup（trigger達成後）') -and
$bridgePlanCompletenessScriptText.Contains('trigger_cleanup_completion_report.json') -and
$bridgePlanCompletenessScriptText.Contains('close_policy_8_1_report.json') -and
$bridgePlanCompletenessScriptText.Contains('close_policy_8_1_report_v1') -and
$bridgePlanCompletenessScriptText.Contains('close_policy_8_1_workflow_input_contract_report.json') -and
$bridgePlanCompletenessScriptText.Contains('close_policy_8_1_workflow_input_contract_report_v1') -and
$bridgePlanCompletenessScriptText.Contains('close_policy_8_1_workflow_input_coherence_report.json') -and
$bridgePlanCompletenessScriptText.Contains('close_policy_8_1_workflow_input_coherence_report_v1') -and
$bridgePlanCompletenessScriptText.Contains('backlog_specfixed_residual_report.json') -and
$bridgePlanCompletenessScriptText.Contains('backlog_specfixed_residual_report_v1') -and
$bridgePlanCompletenessScriptText.Contains('cleanupCompleted=true') -and
$bridgePlanCompletenessScriptText.Contains('deferredRegistryEntryCount=0') -and
$bridgePlanCompletenessScriptText.Contains('deferredRegistryPath mismatch') -and
$bridgePlanCompletenessScriptText.Contains('sourceRoot mismatch') -and
$bridgePlanCompletenessScriptText.Contains('backlogPath mismatch') -and
$bridgePlanCompletenessScriptText.Contains('artifactFreshnessWindowMinutes') -and
$bridgePlanCompletenessScriptText.Contains('freshness breach') -and
$bridgePlanCompletenessScriptText.Contains('generatedAt parse failed') -and
$bridgePlanCompletenessScriptText.Contains('enforceNoSpecFixed=true') -and
$bridgePlanCompletenessScriptText.Contains('specFixedResidualCount=0') -and
$bridgePlanCompletenessScriptText.Contains('scriptOrder') -and
$bridgePlanCompletenessScriptText.Contains('backlogBeforeBridgeCompleteness') -and
$bridgePlanCompletenessScriptText.Contains('backlogResidualForwarding') -and
$bridgePlanCompletenessScriptText.Contains('enforceNoSpecFixedForwarded') -and
$bridgePlanCompletenessScriptText.Contains('isr-8_1-close-policy.json') -and
$bridgePlanCompletenessScriptText.Contains('isr_8_1_close_policy_v1') -and
$bridgePlanCompletenessScriptText.Contains('policyStatus') -and
$bridgePlanCompletenessScriptText.Contains('cleanupReferenceConsistency') -and
$bridgePlanCompletenessScriptText.Contains('deferredRegistryPath') -and
$bridgePlanCompletenessScriptText.Contains('sourceRoot') -and
$bridgePlanCompletenessScriptText.Contains('satisfied')
if (-not $bridgePlanCompletenessNeedsContracts) {
    throw 'Bridge plan completeness gate missing core contract checks'
}

$closePolicyGateScriptPath = Join-Path $repoRoot '.github\scripts\isr-verify-8_1-close-policy.ps1'
if (-not (Test-Path $closePolicyGateScriptPath)) {
    throw "Missing 8.1 close policy gate script: $closePolicyGateScriptPath"
}

$closePolicyGateScriptText = Get-Content -LiteralPath $closePolicyGateScriptPath -Raw -Encoding UTF8
$closePolicyGateNeedsContracts =
$closePolicyGateScriptText.Contains('close_policy_8_1_report.json') -and
$closePolicyGateScriptText.Contains('close_policy_8_1_report_v1') -and
$closePolicyGateScriptText.Contains('isr-8_1-close-policy.json') -and
$closePolicyGateScriptText.Contains('isr_8_1_close_policy_v1') -and
$closePolicyGateScriptText.Contains('workflowInputContract') -and
$closePolicyGateScriptText.Contains('descriptionMustContain') -and
$closePolicyGateScriptText.Contains('expiryGuardDaysByTier') -and
$closePolicyGateScriptText.Contains('expiryDaysRemaining') -and
$closePolicyGateScriptText.Contains('activeTierGuardDays') -and
$closePolicyGateScriptText.Contains('expiry guard breached for tier=') -and
$closePolicyGateScriptText.Contains('allowedCollectTiers') -and
$closePolicyGateScriptText.Contains('allowedEnforceTiers') -and
$closePolicyGateScriptText.Contains('8.1 close policy collector invalid range')
if (-not $closePolicyGateNeedsContracts) {
    throw '8.1 close policy gate missing core contract checks'
}

$closePolicyWorkflowCoherenceGateScriptPath = Join-Path $repoRoot '.github\scripts\isr-verify-8_1-workflow-input-coherence.ps1'
if (-not (Test-Path $closePolicyWorkflowCoherenceGateScriptPath)) {
    throw "Missing 8.1 workflow input coherence gate script: $closePolicyWorkflowCoherenceGateScriptPath"
}

$closePolicyWorkflowContractGateScriptPath = Join-Path $repoRoot '.github\scripts\isr-verify-8_1-workflow-input-contract.ps1'
if (-not (Test-Path $closePolicyWorkflowContractGateScriptPath)) {
    throw "Missing 8.1 workflow input contract gate script: $closePolicyWorkflowContractGateScriptPath"
}

$closePolicyWorkflowContractGateScriptText = Get-Content -LiteralPath $closePolicyWorkflowContractGateScriptPath -Raw -Encoding UTF8
$closePolicyWorkflowContractNeedsContracts =
$closePolicyWorkflowContractGateScriptText.Contains('close_policy_8_1_workflow_input_contract_report.json') -and
$closePolicyWorkflowContractGateScriptText.Contains('close_policy_8_1_workflow_input_contract_report_v1') -and
$closePolicyWorkflowContractGateScriptText.Contains('isr-verification.yml') -and
$closePolicyWorkflowContractGateScriptText.Contains('isr-8_1-close-policy.json') -and
$closePolicyWorkflowContractGateScriptText.Contains('workflowInputContract') -and
$closePolicyWorkflowContractGateScriptText.Contains('descriptionMustContain') -and
$closePolicyWorkflowContractGateScriptText.Contains('Get-WorkflowDispatchInputNames') -and
$closePolicyWorkflowContractGateScriptText.Contains('8.1 close policy workflowInputContract missing required field: inputs') -and
$closePolicyWorkflowContractGateScriptText.Contains('8.1 close policy workflowInputContract has duplicate input name') -and
$closePolicyWorkflowContractGateScriptText.Contains('8.1 close policy workflowInputContract input has invalid type') -and
$closePolicyWorkflowContractGateScriptText.Contains('8.1 close policy workflowInputContract string input missing policy range fields') -and
$closePolicyWorkflowContractGateScriptText.Contains('8.1 close policy workflowInputContract references unknown collector field') -and
$closePolicyWorkflowContractGateScriptText.Contains('Workflow input contract mismatch: uncontracted 8.1 workflow input detected') -and
$closePolicyWorkflowContractGateScriptText.Contains('policyMinField') -and
$closePolicyWorkflowContractGateScriptText.Contains('policyMaxField') -and
$closePolicyWorkflowContractGateScriptText.Contains('type mismatch') -and
$closePolicyWorkflowContractGateScriptText.Contains('required mismatch') -and
$closePolicyWorkflowContractGateScriptText.Contains('default mismatch') -and
$closePolicyWorkflowContractGateScriptText.Contains("description must include '")
if (-not $closePolicyWorkflowContractNeedsContracts) {
    throw '8.1 workflow input contract gate missing core contract checks'
}

$closePolicyWorkflowCoherenceGateScriptText = Get-Content -LiteralPath $closePolicyWorkflowCoherenceGateScriptPath -Raw -Encoding UTF8
$closePolicyWorkflowCoherenceNeedsContracts =
$closePolicyWorkflowCoherenceGateScriptText.Contains('close_policy_8_1_workflow_input_coherence_report.json') -and
$closePolicyWorkflowCoherenceGateScriptText.Contains('close_policy_8_1_workflow_input_coherence_report_v1') -and
$closePolicyWorkflowCoherenceGateScriptText.Contains('isr-verification.yml') -and
$closePolicyWorkflowCoherenceGateScriptText.Contains('isr-8_1-close-policy.json') -and
$closePolicyWorkflowCoherenceGateScriptText.Contains('workflowInputContract') -and
$closePolicyWorkflowCoherenceGateScriptText.Contains('policyMinField') -and
$closePolicyWorkflowCoherenceGateScriptText.Contains('policyMaxField') -and
$closePolicyWorkflowCoherenceGateScriptText.Contains('coherenceInputs') -and
$closePolicyWorkflowCoherenceGateScriptText.Contains('has no string inputs with policy range fields for coherence checks') -and
$closePolicyWorkflowCoherenceGateScriptText.Contains('out of policy range')
if (-not $closePolicyWorkflowCoherenceNeedsContracts) {
    throw '8.1 workflow input coherence gate missing core contract checks'
}

$tierRunnerBacklogScriptToken = "'.github/scripts/isr-verify-backlog-specfixed-residual.ps1'"
$tierRunnerBridgeCompletenessToken = "'.github/scripts/isr-verify-bridge-plan-completeness.ps1'"
$tierRunnerBacklogScriptIndex = $tierRunnerText.IndexOf($tierRunnerBacklogScriptToken)
$tierRunnerBridgeCompletenessIndex = $tierRunnerText.IndexOf($tierRunnerBridgeCompletenessToken)
if ($tierRunnerBacklogScriptIndex -lt 0 -or $tierRunnerBridgeCompletenessIndex -lt 0) {
    throw 'Tier runner missing backlog/bridge completeness script tokens for order self-test'
}

if ($tierRunnerBacklogScriptIndex -ge $tierRunnerBridgeCompletenessIndex) {
    throw 'Tier runner order self-test failed: backlog spec-fixed residual gate must run before bridge plan completeness gate'
}

foreach ($forbiddenPattern in @(
        '\.github/scripts/isr-verify-',
        'validator_tiering_report\.json',
        'trigger_cleanup_completion_report\.json',
        '\bISR_REQUIRE_RUNTIME_EVIDENCE\b'
    )) {
    if (-not $validatorTieringText.Contains($forbiddenPattern)) {
        throw "Validator tiering gate missing forbidden runtime dependency pattern: $forbiddenPattern"
    }
}

$triggerCleanupNeedsFullSrcScan = $triggerCleanupCompletionText.Contains("$sourceRoot = Join-Path `$repoRoot 'src'")
if (-not $triggerCleanupNeedsFullSrcScan) {
    throw 'Trigger cleanup completion gate must scan full src root'
}

if (-not $triggerCleanupCompletionText.Contains('$report.policyEvaluations')) {
    throw 'Trigger cleanup completion gate missing policyEvaluations verification'
}

foreach ($metricField in @(
        'activeDspRefCount',
        'fadingOutDspWriteCount',
        'retireFacadeDirectDependencyCount',
        'retireFacadeRuntimeExecutionCount',
        'runtimeExecutionViewUsageCount',
        'legacyDirectObserveUsageCount'
    )) {
    if (-not $triggerCleanupCompletionText.Contains($metricField)) {
        throw "Trigger cleanup completion gate missing metric field wiring: $metricField"
    }
}

foreach ($legacyHelperName in @(
        'getActiveDSP',
        'resolveActiveDSPFromRuntimeWorldOnly',
        'resolveFadingDSPFromRuntimeWorldOnly',
        'exchangeFadingOutDSP'
    )) {
    if (-not $triggerCleanupCompletionText.Contains($legacyHelperName)) {
        throw "Trigger cleanup completion gate missing legacy helper scan target: $legacyHelperName"
    }
}

if (-not $triggerCleanupCompletionText.Contains('[switch]$RequireAstEvidence')) {
    throw 'Trigger cleanup completion gate missing RequireAstEvidence parameter'
}

if (-not $triggerCleanupCompletionText.Contains('trigger_audit_report.astEvidenceRequired=true')) {
    throw 'Trigger cleanup completion gate missing astEvidenceRequired contract check'
}

if (-not $triggerCleanupCompletionText.Contains('cleanup_deferred_registry_v1')) {
    throw 'Trigger cleanup completion gate missing deferred registry schema contract check'
}

if (-not $triggerCleanupCompletionText.Contains('requires deferred cleanup registry to be empty')) {
    throw 'Trigger cleanup completion gate missing deferred registry emptiness contract check'
}

if (-not $triggerCleanupCompletionText.Contains('referenceConsistency')) {
    throw 'Trigger cleanup completion gate missing referenceConsistency report block'
}

if (-not $triggerCleanupCompletionText.Contains('triggerAuditReportPathSatisfied')) {
    throw 'Trigger cleanup completion gate missing triggerAuditReportPathSatisfied wiring'
}

if (-not $triggerCleanupCompletionText.Contains('deferredRegistryPathSatisfied')) {
    throw 'Trigger cleanup completion gate missing deferredRegistryPathSatisfied wiring'
}

if (-not $triggerCleanupCompletionText.Contains('sourceRootSatisfied')) {
    throw 'Trigger cleanup completion gate missing sourceRootSatisfied wiring'
}

if (-not $triggerCleanupCompletionText.Contains('Trigger cleanup completion reference mismatch: triggerAuditReport')) {
    throw 'Trigger cleanup completion gate missing triggerAuditReport reference mismatch fail-closed wiring'
}

if (-not $triggerCleanupCompletionText.Contains('Trigger cleanup completion reference mismatch: deferredRegistryPath')) {
    throw 'Trigger cleanup completion gate missing deferredRegistryPath reference mismatch fail-closed wiring'
}

if (-not $triggerCleanupCompletionText.Contains('Trigger cleanup completion reference mismatch: sourceRoot')) {
    throw 'Trigger cleanup completion gate missing sourceRoot reference mismatch fail-closed wiring'
}

Write-Host '[PASS] ISR gate wiring self-test verified'
