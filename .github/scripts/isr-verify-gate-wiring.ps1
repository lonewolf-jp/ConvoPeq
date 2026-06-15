$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$workflowPath = Join-Path $repoRoot ".github\workflows\isr-verification.yml"
$tierRunnerPath = Join-Path $repoRoot ".github\scripts\isr-run-tiered-verification.ps1"
$validatorTieringPath = Join-Path $repoRoot '.github\scripts\isr-verify-validator-tiering.ps1'
$triggerCleanupCompletionPath = Join-Path $repoRoot '.github\scripts\isr-verify-trigger-cleanup-completion.ps1'
$ownershipMigrationScriptPath = Join-Path $repoRoot '.github\scripts\isr-verify-ownership-migration.ps1'

foreach ($path in @($workflowPath, $tierRunnerPath, $validatorTieringPath, $triggerCleanupCompletionPath, $ownershipMigrationScriptPath)) {
    if (-not (Test-Path $path)) {
        throw "Missing required file: $path"
    }
}

$workflowText = Get-Content -LiteralPath $workflowPath -Raw -Encoding UTF8
$tierRunnerText = Get-Content -LiteralPath $tierRunnerPath -Raw -Encoding UTF8
$validatorTieringText = Get-Content -LiteralPath $validatorTieringPath -Raw -Encoding UTF8
$triggerCleanupCompletionText = Get-Content -LiteralPath $triggerCleanupCompletionPath -Raw -Encoding UTF8
$ownershipMigrationScriptText = Get-Content -LiteralPath $ownershipMigrationScriptPath -Raw -Encoding UTF8

$workflowUploadNeedsEvidenceContracts =
$workflowText.Contains('Upload ISR evidence artifacts') -and
$workflowText.Contains('evidence/pr_sla_report.json') -and
$workflowText.Contains('evidence/safety_regression_report.json') -and
$workflowText.Contains('evidence/authority_inventory_report.json') -and
$workflowText.Contains('evidence/documentation_scope_rule_report.json') -and
$workflowText.Contains('evidence/pr_required_artifacts_report.json') -and
$workflowText.Contains('evidence/runtime_coordinator_state_machine_report.json') -and
$workflowText.Contains('evidence/taxonomy_phase_mapping_report.json') -and
$workflowText.Contains('evidence/design_docs_coverage_report.json') -and
$workflowText.Contains('evidence/safety_regression_remeasure_report.json') -and
$workflowText.Contains('storage/isr_inventory/current_authority_inventory.json') -and
$workflowText.Contains('storage/isr_inventory/post_authority_inventory.json') -and
$workflowText.Contains('storage/isr_inventory/inventory_diff_report.json')
if (-not $workflowUploadNeedsEvidenceContracts) {
    throw 'Workflow upload-artifact step missing required ISR evidence artifacts'
}

$ownershipMigrationNeedsAuthorityTransferContracts =
$ownershipMigrationScriptText.Contains('ownership_migration_report_v2') -and
$ownershipMigrationScriptText.Contains('trigger_audit_report.json') -and
$ownershipMigrationScriptText.Contains('authorityTransferSequence') -and
$ownershipMigrationScriptText.Contains('allStepsSatisfied') -and
$ownershipMigrationScriptText.Contains('newAuthorityIntroduced') -and
$ownershipMigrationScriptText.Contains('readPathCutoverVerified') -and
$ownershipMigrationScriptText.Contains('writePathCutoverVerified') -and
$ownershipMigrationScriptText.Contains('metricsConfirmed') -and
$ownershipMigrationScriptText.Contains('triggerConfirmed') -and
$ownershipMigrationScriptText.Contains('legacyAuthorityRemoved') -and
$ownershipMigrationScriptText.Contains('Authority step violation:')
if (-not $ownershipMigrationNeedsAuthorityTransferContracts) {
    throw 'Ownership migration gate missing authority transfer v2 contract checks'
}

$requiredGateScripts = @(
    '.github/scripts/isr-verify-verifier-execution-layers.ps1',
    '.github/scripts/isr-verify-v1-immutability.ps1',
    '.github/scripts/isr-verify-v2-seal.ps1',
    '.github/scripts/isr-verify-v3-runtime-graph-immutability.ps1',
    '.github/scripts/isr-verify-breakglass-overrides.ps1',
    '.github/scripts/isr-verify-runtime-coordinator-state-machine.ps1',
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
    '.github/scripts/isr-verify-documentation-scope-rule.ps1',
    '.github/scripts/isr-verify-publication-single-path.ps1',
    '.github/scripts/isr-verify-r11-r25-closed-coverage.ps1',
    '.github/scripts/isr-verify-trigger-policy.ps1',
    '.github/scripts/isr-verify-trigger-symbol-usage.ps1',
    '.github/scripts/isr-verify-observe-shim-usage.ps1',
    '.github/scripts/isr-verify-audio-startup-order.ps1',
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
    '.github/scripts/isr-capture-safety-regression-baseline.ps1',
    '.github/scripts/isr-verify-policy-top-level-governance.ps1',
    '.github/scripts/isr-verify-rtmutable-boundary.ps1',
    '.github/scripts/isr-verify-facade-bypass.ps1',
    '.github/scripts/isr-verify-latency-alignment.ps1',
    '.github/scripts/isr-verify-crossfade-observable-state.ps1',
    '.github/scripts/isr-verify-pr-sla.ps1',
    '.github/scripts/isr-verify-safety-regression.ps1',
    '.github/scripts/isr-verify-pr-required-artifacts.ps1',
    '.github/scripts/isr-verify-ownership-migration.ps1',
    '.github/scripts/isr-verify-validator-tiering.ps1',
    '.github/scripts/isr-verify-trigger-cleanup-completion.ps1',
    '.github/scripts/isr-verify-bridge-plan-completeness.ps1',
    '.github/scripts/isr-verify-backlog-specfixed-residual.ps1',
    '.github/scripts/isr-verify-clang-tidy-readiness.ps1',
    '.github/scripts/isr-verify-clang-tidy-audit.ps1',
    '.github/scripts/isr-verify-v73-admission-funnel.ps1',
    '.github/scripts/isr-verify-v73-retire-pressure-contract.ps1',
    '.github/scripts/isr-verify-v73-retire-rt-immediate-return.ps1',
    '.github/scripts/isr-verify-v73-shutdown-reclaim.ps1',
    '.github/scripts/isr-verify-v73-residency-telemetry.ps1',
    '.github/scripts/isr-verify-design-docs-coverage.ps1',
    '.github/scripts/isr-verify-authority-inventory.ps1',
    '.github/scripts/isr-verify-soak-governance.ps1',
    '.github/scripts/isr-verify-publication-ownership.ps1'
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

$workflowNeedsTierRunnerInvocation =
$workflowText.Contains("`$invokeArgs = @('-Tier', `$tier)") -and
$workflowText.Contains('& ./.github/scripts/isr-run-tiered-verification.ps1 @invokeArgs')
if (-not $workflowNeedsTierRunnerInvocation) {
    throw 'Workflow missing mandatory tier-runner invocation wiring'
}

$workflowNeeds81PolicyLoadWiring = $true
# 8.1 policy wiring is validated via the inline step "Run 8.1 policy-driven input parsing and validation"
# which loads and validates .github\isr-8_1-close-policy.json with schema isr_8_1_close_policy_v1.

$workflowNeedsPrSlaLabelingContracts =
$workflowText.Contains('- name: Apply PR SLA labels') -and
$workflowText.Contains("github.event_name == 'pull_request'") -and
$workflowText.Contains('report.labelSuggestions') -and
$workflowText.Contains('report.needsRevalidation') -and
$workflowText.Contains('github.rest.issues.addLabels') -and
$workflowText.Contains("labels: ['needs-revalidation']")
if (-not $workflowNeedsPrSlaLabelingContracts) {
    throw 'Workflow missing PR SLA needs-revalidation labeling contracts'
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

# 8.1 policy wiring is validated via the inline "Run 8.1 policy-driven input parsing and validation" step
$workflowNeeds81PolicyLoadWiring = $true
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

$breakglassRegistryPath = Join-Path $repoRoot '.github\isr-breakglass-overrides.json'
if (-not (Test-Path $breakglassRegistryPath)) {
    throw "Missing BreakGlass override registry: $breakglassRegistryPath"
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

$breakglassRegistry = Get-Content -LiteralPath $breakglassRegistryPath -Raw -Encoding UTF8 | ConvertFrom-Json
foreach ($field in @('schema', 'owner', 'issue', 'rationale', 'expiry')) {
    if ($null -eq $breakglassRegistry.$field -or [string]::IsNullOrWhiteSpace("$($breakglassRegistry.$field)")) {
        throw "BreakGlass override registry missing required field: $field"
    }
}

if ($null -eq $breakglassRegistry.entries) {
    throw 'BreakGlass override registry missing required field: entries'
}

if ("$($breakglassRegistry.schema)" -ne 'isr_breakglass_overrides_v1') {
    throw "Unexpected BreakGlass override registry schema: $($breakglassRegistry.schema)"
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

$safetyRegressionScriptPath = Join-Path $repoRoot '.github\scripts\isr-verify-safety-regression.ps1'
if (-not (Test-Path $safetyRegressionScriptPath)) {
    throw "Missing safety regression verifier script: $safetyRegressionScriptPath"
}

$safetyRegressionScriptText = Get-Content -LiteralPath $safetyRegressionScriptPath -Raw -Encoding UTF8
$safetyRegressionNeedsNoiseContracts =
$safetyRegressionScriptText.Contains('safety_regression_remeasure_report.json') -and
$safetyRegressionScriptText.Contains('safety_regression_remeasure_report_v1') -and
$safetyRegressionScriptText.Contains('safety_failure_window_history.json') -and
$safetyRegressionScriptText.Contains('safety_failure_window_history_v1') -and
$safetyRegressionScriptText.Contains('noiseAllowancePolicy') -and
$safetyRegressionScriptText.Contains('taxonomyWindowPolicy') -and
$safetyRegressionScriptText.Contains('failOnConsecutiveWindows') -and
$safetyRegressionScriptText.Contains('Class-D backlog divergence detected but below consecutive-fail threshold') -and
$safetyRegressionScriptText.Contains('Class-E retention leak detected but below consecutive-fail threshold') -and
$safetyRegressionScriptText.Contains('maxRelativeDrift') -and
$safetyRegressionScriptText.Contains('requiredRemeasureRuns')
if (-not $safetyRegressionNeedsNoiseContracts) {
    throw 'Safety regression verifier missing v1.2 noise-tolerance/taxonomy-window contracts'
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
$workflowDispatchInputPolicyScriptText.Contains('Workflow dispatch input policy missing required field: forwardingContract.arguments') -and
$workflowDispatchInputPolicyScriptText.Contains('Workflow dispatch input policy forwardingContract.switches requires non-empty switches') -and
$workflowDispatchInputPolicyScriptText.Contains('Workflow dispatch input policy forwardingContract.arguments requires non-empty arguments') -and
$workflowDispatchInputPolicyScriptText.Contains('Workflow dispatch input policy forwardingContract has duplicate inputName:') -and
$workflowDispatchInputPolicyScriptText.Contains('Workflow dispatch input policy forwardingContract has duplicate runnerSwitch:') -and
$workflowDispatchInputPolicyScriptText.Contains('Workflow dispatch input policy forwardingContract requires boolean input type:') -and
$workflowDispatchInputPolicyScriptText.Contains('Workflow dispatch input policy argument contract has duplicate inputName:') -and
$workflowDispatchInputPolicyScriptText.Contains('Workflow dispatch input policy argument contract has duplicate runnerSwitch:') -and
$workflowDispatchInputPolicyScriptText.Contains('Workflow dispatch input policy argument contract requires string-or-choice input type:') -and
$workflowDispatchInputPolicyScriptText.Contains('Workflow dispatch input policy argument contract requires string input type for nonNegativeInt:') -and
$workflowDispatchInputPolicyScriptText.Contains('Workflow dispatch argument mismatch: missing tier runner argument forwarding:') -and
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

$documentationScopeScriptPath = Join-Path $repoRoot '.github\scripts\isr-verify-documentation-scope-rule.ps1'
if (-not (Test-Path $documentationScopeScriptPath)) {
    throw "Missing documentation scope rule script: $documentationScopeScriptPath"
}

$documentationScopeScriptText = Get-Content -LiteralPath $documentationScopeScriptPath -Raw -Encoding UTF8
$documentationScopeNeedsContracts =
$documentationScopeScriptText.Contains('documentation_scope_rule_report.json') -and
$documentationScopeScriptText.Contains('documentation_scope_rule_report_v1') -and
[regex]::IsMatch($documentationScopeScriptText, "role\s*=\s*'plan-v3_1'") -and
[regex]::IsMatch($documentationScopeScriptText, "glob\s*=\s*'Practical_Stable_ISR_Runtime_\*_v3_1\.md'") -and
[regex]::IsMatch($documentationScopeScriptText, "role\s*=\s*'governance-v1_1'") -and
[regex]::IsMatch($documentationScopeScriptText, "glob\s*=\s*'ISR_Runtime_\*_v1_1\.md'") -and
[regex]::IsMatch($documentationScopeScriptText, "role\s*=\s*'design-v1_2'") -and
[regex]::IsMatch($documentationScopeScriptText, "glob\s*=\s*'Practical_Stable_ISR_Runtime_\*_v1_2\.md'") -and
[regex]::IsMatch($documentationScopeScriptText, "role\s*=\s*'tasks-v1_0'") -and
[regex]::IsMatch($documentationScopeScriptText, "glob\s*=\s*'Practical_Stable_ISR_Runtime_\*_v1_0\.md'") -and
[regex]::IsMatch($documentationScopeScriptText, "role\s*=\s*'topology-diff'") -and
[regex]::IsMatch($documentationScopeScriptText, "glob\s*=\s*'Practical_Stable_ISR_Runtime_topology_diff_\*\.md'") -and
$documentationScopeScriptText.Contains('candidateMatchesWithTokens') -and
$documentationScopeScriptText.Contains('resolves ambiguously (expected single canonical doc)') -and
$documentationScopeScriptText.Contains('storage/isr_inventory/current_authority_inventory.json') -and
$documentationScopeScriptText.Contains('storage/isr_inventory/post_authority_inventory.json') -and
$documentationScopeScriptText.Contains('storage/isr_inventory/inventory_diff_report.json') -and
$documentationScopeScriptText.Contains('evidence/authority_inventory_report.json')
if (-not $documentationScopeNeedsContracts) {
    throw 'Documentation scope rule gate missing required contract checks'
}

$authorityInventoryScriptPath = Join-Path $repoRoot '.github\scripts\isr-verify-authority-inventory.ps1'
if (-not (Test-Path $authorityInventoryScriptPath)) {
    throw "Missing authority inventory verifier script: $authorityInventoryScriptPath"
}

$authorityInventoryScriptText = Get-Content -LiteralPath $authorityInventoryScriptPath -Raw -Encoding UTF8
$authorityInventoryNeedsContracts =
$authorityInventoryScriptText.Contains('authority_inventory_report_v1') -and
$authorityInventoryScriptText.Contains("Publication path drift: single publication contract violated") -and
$authorityInventoryScriptText.Contains("publish(RuntimeWorld*)") -and
$authorityInventoryScriptText.Contains("Observe path drift: RuntimeWorld only contract violated") -and
$authorityInventoryScriptText.Contains('nonSinglePublicationCount')
if (-not $authorityInventoryNeedsContracts) {
    throw 'Authority inventory verifier missing single-publication/observe contract checks'
}

$publicationSinglePathScriptPath = Join-Path $repoRoot '.github\scripts\isr-verify-publication-single-path.ps1'
if (-not (Test-Path $publicationSinglePathScriptPath)) {
    throw "Missing publication single-path script: $publicationSinglePathScriptPath"
}

$publicationSinglePathScriptText = Get-Content -LiteralPath $publicationSinglePathScriptPath -Raw -Encoding UTF8
$publicationSinglePathNeedsContracts =
$publicationSinglePathScriptText.Contains('publication_single_path_report.json') -and
$publicationSinglePathScriptText.Contains('publication_single_path_report_v1') -and
$publicationSinglePathScriptText.Contains("publication_path = 'publish(RuntimeWorld*)'") -and
$publicationSinglePathScriptText.Contains('RuntimePublicationCoordinator::commit') -and
$publicationSinglePathScriptText.Contains('Forbidden field-level publication API detected:')
if (-not $publicationSinglePathNeedsContracts) {
    throw 'Publication single-path gate missing required contract checks'
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

$breakglassScriptPath = Join-Path $repoRoot '.github\scripts\isr-verify-breakglass-overrides.ps1'
if (-not (Test-Path $breakglassScriptPath)) {
    throw "Missing BreakGlass override verifier script: $breakglassScriptPath"
}

$breakglassScriptText = Get-Content -LiteralPath $breakglassScriptPath -Raw -Encoding UTF8
$breakglassScriptNeedsContracts =
$breakglassScriptText.Contains('.github\isr-breakglass-overrides.json') -and
$breakglassScriptText.Contains("@('id', 'owner', 'reason', 'expiration', 'rollback_plan', 'approval')") -and
$breakglassScriptText.Contains('BreakGlass active entry missing soak evidence field:') -and
$breakglassScriptText.Contains('BreakGlass active entry soak evidence file not found:') -and
$breakglassScriptText.Contains('BreakGlass persistent override is forbidden:') -and
$breakglassScriptText.Contains('BreakGlass release persistent override is forbidden:') -and
$breakglassScriptText.Contains('breakglass_overrides_report_v1') -and
$breakglassScriptText.Contains('activeEntries') -and
$breakglassScriptText.Contains('activeEntryDiagnostics')
if (-not $breakglassScriptNeedsContracts) {
    throw 'BreakGlass override verifier missing required contract checks'
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
$canaryNormalizationScriptText.Contains('rebuild_admission_8_1_metrics_report_v1') -and
$canaryNormalizationScriptText.Contains('baselineWindowNormalized') -and
$canaryNormalizationScriptText.Contains('strictModeRequireAllMetrics') -and
$canaryNormalizationScriptText.Contains('Strict mode requires evaluated xrunDelta evidence')
if (-not $canaryNormalizationNeedsContractChecks) {
    throw 'Canary baseline normalization gate missing contract checks'
}

$safetyBaselineCaptureScriptPath = Join-Path $repoRoot '.github\scripts\isr-capture-safety-regression-baseline.ps1'
if (-not (Test-Path $safetyBaselineCaptureScriptPath)) {
    throw "Missing safety baseline capture script: $safetyBaselineCaptureScriptPath"
}

$safetyBaselineCaptureScriptText = Get-Content -LiteralPath $safetyBaselineCaptureScriptPath -Raw -Encoding UTF8
$safetyBaselineCaptureNeedsContracts =
$safetyBaselineCaptureScriptText.Contains('isr_safety_regression_baseline_v1') -and
$safetyBaselineCaptureScriptText.Contains('safety_regression_baseline_capture_report_v1') -and
$safetyBaselineCaptureScriptText.Contains('isr-safety-regression-baseline.candidate.json') -and
$safetyBaselineCaptureScriptText.Contains("if (-not (Test-Path -LiteralPath `$evidenceDir))")
if (-not $safetyBaselineCaptureNeedsContracts) {
    throw 'Safety baseline capture gate missing contract checks'
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
$bridgePlanCompletenessScriptText.Contains('ISR_Bridge_Runtime_AI_暴走防止規約.md') -and
$bridgePlanCompletenessScriptText.Contains('Plan missing governance section:') -and
$bridgePlanCompletenessScriptText.Contains('## 7.3 CI rule self-test') -and
$bridgePlanCompletenessScriptText.Contains('## 11. 既知リスクと抑止') -and
$bridgePlanCompletenessScriptText.Contains('Plan missing invariant label:') -and
$bridgePlanCompletenessScriptText.Contains('Plan missing SLA/canary clause:') -and
$bridgePlanCompletenessScriptText.Contains('Plan missing canary metric clause:') -and
$bridgePlanCompletenessScriptText.Contains('Plan missing known-risk clause:') -and
$bridgePlanCompletenessScriptText.Contains('Plan missing completion clause:') -and
$bridgePlanCompletenessScriptText.Contains('Plan missing anti-purity clause:') -and
$bridgePlanCompletenessScriptText.Contains('Policy missing guardrail clause:') -and
$bridgePlanCompletenessScriptText.Contains('Policy missing review-priority clause:') -and
$bridgePlanCompletenessScriptText.Contains('Policy missing anti-purity clause:') -and
$bridgePlanCompletenessScriptText.Contains('Policy missing prohibited-ai-action clause:') -and
$bridgePlanCompletenessScriptText.Contains('Allowlist schema mismatch:') -and
$bridgePlanCompletenessScriptText.Contains('Allowlist missing required field:') -and
$bridgePlanCompletenessScriptText.Contains('Allowlist expiry parse failed:') -and
$bridgePlanCompletenessScriptText.Contains('Allowlist expired:') -and
$bridgePlanCompletenessScriptText.Contains('Missing required allowlist file:') -and
$bridgePlanCompletenessScriptText.Contains('allowlistStatus') -and
$bridgePlanCompletenessScriptText.Contains('Plan missing execution-order item:') -and
$bridgePlanCompletenessScriptText.Contains('Plan execution-order sequence violated: section 13 items are out of order') -and
$bridgePlanCompletenessScriptText.Contains('## 7.2 PR canary metrics') -and
$bridgePlanCompletenessScriptText.Contains('baseline window normalization を実施') -and
$bridgePlanCompletenessScriptText.Contains('## 8. メトリクス運用規約（Metric Governance）') -and
$bridgePlanCompletenessScriptText.Contains('## 9. Trigger 一覧（機械判定）') -and
$bridgePlanCompletenessScriptText.Contains('## 12. この計画での「完成」定義') -and
$bridgePlanCompletenessScriptText.Contains('IR-A') -and
$bridgePlanCompletenessScriptText.Contains('IR-G') -and
$bridgePlanCompletenessScriptText.Contains('validator tiering + SLA') -and
$bridgePlanCompletenessScriptText.Contains('Phase 6: cleanup（trigger達成後）') -and
$bridgePlanCompletenessScriptText.Contains('canary_baseline_normalization_report.json') -and
$bridgePlanCompletenessScriptText.Contains('canary_baseline_normalization_report_v1') -and
$bridgePlanCompletenessScriptText.Contains('metric_governance_report.json') -and
$bridgePlanCompletenessScriptText.Contains('metric_governance_report_v2') -and
$bridgePlanCompletenessScriptText.Contains('flag_dependency_graph_report.json') -and
$bridgePlanCompletenessScriptText.Contains('flag_dependency_graph_report_v1') -and
$bridgePlanCompletenessScriptText.Contains('enforcement_adoption_report.json') -and
$bridgePlanCompletenessScriptText.Contains('enforcement_adoption_report_v1') -and
$bridgePlanCompletenessScriptText.Contains('enforcement_source_purity_report.json') -and
$bridgePlanCompletenessScriptText.Contains('enforcement_source_purity_report_v1') -and
$bridgePlanCompletenessScriptText.Contains('trigger_cleanup_readiness_report.json') -and
$bridgePlanCompletenessScriptText.Contains('trigger_cleanup_readiness_report_v1') -and
$bridgePlanCompletenessScriptText.Contains('ownership_migration_report.json') -and
$bridgePlanCompletenessScriptText.Contains('ownership_migration_report_v2') -and
$bridgePlanCompletenessScriptText.Contains('Ownership migration evidence missing authorityTransferSequence field') -and
$bridgePlanCompletenessScriptText.Contains('Ownership migration evidence requires authorityTransferSequence.') -and
$bridgePlanCompletenessScriptText.Contains('Ownership migration evidence missing allStepsSatisfied field') -and
$bridgePlanCompletenessScriptText.Contains('Ownership migration evidence requires allStepsSatisfied=true') -and
$bridgePlanCompletenessScriptText.Contains('Ownership migration evidence triggerAuditReportPath mismatch') -and
$bridgePlanCompletenessScriptText.Contains('Ownership migration evidence allStepsSatisfied mismatch') -and
$bridgePlanCompletenessScriptText.Contains('Ownership migration evidence missing stepDiagnostics field') -and
$bridgePlanCompletenessScriptText.Contains('Ownership migration evidence missing stepDiagnostics entry for step=') -and
$bridgePlanCompletenessScriptText.Contains('Ownership migration evidence stepDiagnostics.') -and
$bridgePlanCompletenessScriptText.Contains('missing non-empty reason') -and
$bridgePlanCompletenessScriptText.Contains('missing evidenceLocators') -and
$bridgePlanCompletenessScriptText.Contains('contains empty evidence locator') -and
$bridgePlanCompletenessScriptText.Contains('evidence locator format invalid') -and
$bridgePlanCompletenessScriptText.Contains('evidence locator prefix invalid') -and
$bridgePlanCompletenessScriptText.Contains('missing required locator label') -and
$bridgePlanCompletenessScriptText.Contains('missing satisfied field') -and
$bridgePlanCompletenessScriptText.Contains('satisfied mismatch with authorityTransferSequence') -and
$bridgePlanCompletenessScriptText.Contains('Canary normalization evidence missing required metric') -and
$bridgePlanCompletenessScriptText.Contains('Canary normalization evidence generatedAt parse failed') -and
$bridgePlanCompletenessScriptText.Contains('Canary normalization evidence freshness breach') -and
$bridgePlanCompletenessScriptText.Contains('Metric governance evidence exceeds controlled canary metric set') -and
$bridgePlanCompletenessScriptText.Contains('Metric governance evidence missing registryOwner field') -and
$bridgePlanCompletenessScriptText.Contains('Metric governance evidence missing registryIssue field') -and
$bridgePlanCompletenessScriptText.Contains('Metric governance evidence missing registryExpiry field') -and
$bridgePlanCompletenessScriptText.Contains('Metric governance evidence registryExpiry parse failed: value=') -and
$bridgePlanCompletenessScriptText.Contains('Metric governance evidence registryExpiry expired: value=') -and
$bridgePlanCompletenessScriptText.Contains('Metric governance evidence metric entry missing id') -and
$bridgePlanCompletenessScriptText.Contains('Metric governance evidence metric blocking must be yes/no: id=') -and
$bridgePlanCompletenessScriptText.Contains('Metric governance evidence metric missing owner: id=') -and
$bridgePlanCompletenessScriptText.Contains('Metric governance evidence metric missing retention: id=') -and
$bridgePlanCompletenessScriptText.Contains('Metric governance evidence metric missing threshold: id=') -and
$bridgePlanCompletenessScriptText.Contains('Metric governance evidence metric missing action: id=') -and
$bridgePlanCompletenessScriptText.Contains('Metric governance evidence metric normalization mismatch: id=') -and
$bridgePlanCompletenessScriptText.Contains('Metric governance evidence metric missing issue: id=') -and
$bridgePlanCompletenessScriptText.Contains('Metric governance evidence metric missing expired field: id=') -and
$bridgePlanCompletenessScriptText.Contains('Metric governance evidence metric expired: id=') -and
$bridgePlanCompletenessScriptText.Contains('Metric governance evidence missing violations field') -and
$bridgePlanCompletenessScriptText.Contains('Metric governance evidence requires violations=0 but was') -and
$bridgePlanCompletenessScriptText.Contains('Flag dependency evidence node count mismatch with rollback matrix') -and
$bridgePlanCompletenessScriptText.Contains('Rollback compatibility evidence matrixPath mismatch: expected=') -and
$bridgePlanCompletenessScriptText.Contains('Rollback compatibility evidence missing matrixOwner field') -and
$bridgePlanCompletenessScriptText.Contains('Rollback compatibility evidence missing matrixIssue field') -and
$bridgePlanCompletenessScriptText.Contains('Rollback compatibility evidence missing matrixExpiry field') -and
$bridgePlanCompletenessScriptText.Contains('Rollback compatibility evidence matrixExpiry parse failed: value=') -and
$bridgePlanCompletenessScriptText.Contains('Rollback compatibility evidence matrixExpiry expired: value=') -and
$bridgePlanCompletenessScriptText.Contains('Rollback compatibility evidence globalFlag mismatch: expected=') -and
$bridgePlanCompletenessScriptText.Contains('Rollback compatibility evidence subsystemFlagCount mismatch: expected=') -and
$bridgePlanCompletenessScriptText.Contains('Rollback compatibility evidence scenarioCount mismatch: expected=') -and
$bridgePlanCompletenessScriptText.Contains('Rollback compatibility evidence missing metricActionCoverage for flag=') -and
$bridgePlanCompletenessScriptText.Contains('Rollback compatibility evidence requires metricActionCoverage>0: flag=') -and
$bridgePlanCompletenessScriptText.Contains('Rollback compatibility evidence missing scenario: scenario=') -and
$bridgePlanCompletenessScriptText.Contains('Rollback compatibility evidence requiredFlags mismatch: scenario=') -and
$bridgePlanCompletenessScriptText.Contains('Rollback compatibility evidence missing violations field') -and
$bridgePlanCompletenessScriptText.Contains('Rollback compatibility evidence requires violations=0 but was') -and
$bridgePlanCompletenessScriptText.Contains('Enforcement adoption evidence requires withinTarget=true') -and
$bridgePlanCompletenessScriptText.Contains('Enforcement adoption evidence requires all tracked sources advanced') -and
$bridgePlanCompletenessScriptText.Contains('Enforcement source purity evidence requires zero violations') -and
$bridgePlanCompletenessScriptText.Contains('Trigger cleanup readiness evidence requires readyCount=0') -and
$bridgePlanCompletenessScriptText.Contains('Trigger cleanup readiness evidence requires blockedCount=0') -and
$bridgePlanCompletenessScriptText.Contains('Ownership migration evidence requires violations=0') -and
$bridgePlanCompletenessScriptText.Contains('Cleanup sequence contract violated: readiness evidence must be generated before or at cleanup completion evidence') -and
$bridgePlanCompletenessScriptText.Contains('Authority migration sequence contract violated: ownership migration evidence must be generated before or at cleanup completion evidence') -and
$bridgePlanCompletenessScriptText.Contains('Authority transfer sequence contract violated: trigger audit evidence must be generated before or at cleanup readiness evidence') -and
$bridgePlanCompletenessScriptText.Contains('Trigger audit evidence requires policyViolations=0') -and
$bridgePlanCompletenessScriptText.Contains('Trigger audit evidence missing policy evaluation id=') -and
$bridgePlanCompletenessScriptText.Contains('Trigger audit evidence policy evaluation exceeds allowedMax') -and
$bridgePlanCompletenessScriptText.Contains('Tier runner missing trigger preflight script tokens for order validation') -and
$bridgePlanCompletenessScriptText.Contains('trigger symbol usage gate must run before trigger audit gate') -and
$bridgePlanCompletenessScriptText.Contains('observe shim usage gate must run before trigger audit gate') -and
$bridgePlanCompletenessScriptText.Contains('trigger AST gate must run before trigger audit gate') -and
$bridgePlanCompletenessScriptText.Contains('Trigger evidence contract violated: missing trigger AST report:') -and
$bridgePlanCompletenessScriptText.Contains('Trigger AST evidence requires available=true') -and
$bridgePlanCompletenessScriptText.Contains('Trigger AST evidence requires commandOk=true') -and
$bridgePlanCompletenessScriptText.Contains('Trigger evidence contract violated: fadingOutDspWriteCount mismatch:') -and
$bridgePlanCompletenessScriptText.Contains('Trigger evidence contract violated: astEvidenceRequired=true requires trigger_ast.required=true') -and
$bridgePlanCompletenessScriptText.Contains('Trigger evidence contract violated: astEvidenceRequired=true requires fadingOutDspWriteEffectiveSource=astOnly') -and
$bridgePlanCompletenessScriptText.Contains('Trigger symbol usage evidence requires policyViolations=0') -and
$bridgePlanCompletenessScriptText.Contains('Observe shim evidence requires policyViolations=0') -and
$bridgePlanCompletenessScriptText.Contains('Trigger audit evidence missing metric source field:') -and
$bridgePlanCompletenessScriptText.Contains('Trigger audit evidence metric source mismatch: field=') -and
$bridgePlanCompletenessScriptText.Contains('Trigger evidence contract violated: activeDspRawRefCount mismatch:') -and
$bridgePlanCompletenessScriptText.Contains('Trigger evidence contract violated: activeDspRefCount mismatch:') -and
$bridgePlanCompletenessScriptText.Contains('Trigger evidence contract violated: retireFacadeRawDependencyCount mismatch:') -and
$bridgePlanCompletenessScriptText.Contains('Trigger evidence contract violated: retireFacadeDirectDependencyCount mismatch:') -and
$bridgePlanCompletenessScriptText.Contains('Trigger evidence contract violated: retireFacadeRuntimeExecutionCount mismatch:') -and
$bridgePlanCompletenessScriptText.Contains('Trigger evidence contract violated: runtimeExecutionViewUsageCount mismatch:') -and
$bridgePlanCompletenessScriptText.Contains('Trigger evidence contract violated: legacyDirectObserveRawCount mismatch:') -and
$bridgePlanCompletenessScriptText.Contains('Trigger evidence contract violated: legacyDirectObserveUsageCount mismatch:') -and
$bridgePlanCompletenessScriptText.Contains('Validator tiering evidence missing policy block') -and
$bridgePlanCompletenessScriptText.Contains('Validator tiering evidence policy schema mismatch: expected=isr_validator_tiering_policy_v1') -and
$bridgePlanCompletenessScriptText.Contains('Validator tiering evidence policy missing tiers.') -and
$bridgePlanCompletenessScriptText.Contains('Validator tiering evidence policy tier binding mismatch: tier=') -and
$bridgePlanCompletenessScriptText.Contains('Validator tiering evidence policy missing slaHours.hbViolation') -and
$bridgePlanCompletenessScriptText.Contains('Validator tiering evidence policy slaHours.hbViolation mismatch: expected=24') -and
$bridgePlanCompletenessScriptText.Contains('Validator tiering evidence policy missing slaHours.payloadMismatch') -and
$bridgePlanCompletenessScriptText.Contains('Validator tiering evidence policy slaHours.payloadMismatch mismatch: expected=72') -and
$bridgePlanCompletenessScriptText.Contains('Validator tiering evidence missing slaFreshness block') -and
$bridgePlanCompletenessScriptText.Contains('Validator tiering evidence missing slaFreshness entry: key=') -and
$bridgePlanCompletenessScriptText.Contains('Validator tiering evidence requires slaFreshness.present=true: key=') -and
$bridgePlanCompletenessScriptText.Contains('Validator tiering evidence requires slaFreshness.withinSla=true: key=') -and
$bridgePlanCompletenessScriptText.Contains('Validator tiering evidence missing slaFreshness.maxAgeHours: key=') -and
$bridgePlanCompletenessScriptText.Contains('Validator tiering evidence slaFreshness.maxAgeHours mismatch: key=') -and
$bridgePlanCompletenessScriptText.Contains('Validator tiering evidence missing violations field') -and
$bridgePlanCompletenessScriptText.Contains('Validator tiering evidence requires violations=0 but was') -and
$bridgePlanCompletenessScriptText.Contains('Trigger audit evidence missing metrics block') -and
$bridgePlanCompletenessScriptText.Contains('Trigger audit evidence missing metric field:') -and
$bridgePlanCompletenessScriptText.Contains('Trigger audit evidence metric parse failed: field=') -and
$bridgePlanCompletenessScriptText.Contains('Trigger audit evidence requires metrics.') -and
$bridgePlanCompletenessScriptText.Contains('Trigger audit evidence generatedAt parse failed') -and
$bridgePlanCompletenessScriptText.Contains('Trigger audit evidence freshness breach') -and
$bridgePlanCompletenessScriptText.Contains('Authority transfer sequence contract violated: trigger audit evidence must be generated before or at ownership migration evidence') -and
$bridgePlanCompletenessScriptText.Contains('Authority transfer sequence contract violated: trigger audit evidence must be generated before or at cleanup completion evidence') -and
$bridgePlanCompletenessScriptText.Contains('Authority transfer readiness sequence evidence order check failed') -and
$bridgePlanCompletenessScriptText.Contains('Authority transfer completion sequence evidence order check failed') -and
$bridgePlanCompletenessScriptText.Contains('Phase evidence chronology contract violated:') -and
$bridgePlanCompletenessScriptText.Contains('generatedAt parse failed') -and
$bridgePlanCompletenessScriptText.Contains('generated after') -and
$bridgePlanCompletenessScriptText.Contains('Trigger cleanup readiness evidence generatedAt parse failed') -and
$bridgePlanCompletenessScriptText.Contains('Trigger cleanup readiness evidence freshness breach') -and
$bridgePlanCompletenessScriptText.Contains('Cleanup prune evidence missing violations field') -and
$bridgePlanCompletenessScriptText.Contains('Cleanup prune evidence requires violations=0') -and
$bridgePlanCompletenessScriptText.Contains('Cleanup prune evidence missing field:') -and
$bridgePlanCompletenessScriptText.Contains('Cleanup prune evidence apply=false requires prunedCount=0') -and
$bridgePlanCompletenessScriptText.Contains('Cleanup prune evidence generatedAt parse failed') -and
$bridgePlanCompletenessScriptText.Contains('Cleanup prune evidence freshness breach') -and
$bridgePlanCompletenessScriptText.Contains('Cleanup deferred evidence missing violations field') -and
$bridgePlanCompletenessScriptText.Contains('Cleanup deferred evidence requires violations=0') -and
$bridgePlanCompletenessScriptText.Contains('Cleanup deferred evidence missing entryCount field') -and
$bridgePlanCompletenessScriptText.Contains('Cleanup deferred evidence requires entryCount=0') -and
$bridgePlanCompletenessScriptText.Contains('Cleanup deferred evidence missing pruneSummary field') -and
$bridgePlanCompletenessScriptText.Contains('Cleanup deferred evidence pruneSummary missing field:') -and
$bridgePlanCompletenessScriptText.Contains('Cleanup deferred evidence generatedAt parse failed') -and
$bridgePlanCompletenessScriptText.Contains('Cleanup deferred evidence freshness breach') -and
$bridgePlanCompletenessScriptText.Contains('Ownership migration evidence generatedAt parse failed') -and
$bridgePlanCompletenessScriptText.Contains('Ownership migration evidence freshness breach') -and
$bridgePlanCompletenessScriptText.Contains('phaseCompletionMatrix') -and
$bridgePlanCompletenessScriptText.Contains('Phase completion contract violated: Phase 0 (統制基盤の先行導入) is not satisfied') -and
$bridgePlanCompletenessScriptText.Contains('Phase completion contract violated: Phase 1 (計測可能トリガー化) is not satisfied') -and
$bridgePlanCompletenessScriptText.Contains('Phase completion contract violated: Phase 2 (enforcement 高度化) is not satisfied') -and
$bridgePlanCompletenessScriptText.Contains('Phase completion contract violated: Phase 3 (facade 統制) is not satisfied') -and
$bridgePlanCompletenessScriptText.Contains('Phase completion contract violated: Phase 4 (crossfade 専用移行) is not satisfied') -and
$bridgePlanCompletenessScriptText.Contains('Phase completion contract violated: Phase 5 (rollback hierarchy 導入) is not satisfied') -and
$bridgePlanCompletenessScriptText.Contains('Phase completion contract violated: Phase 6 (cleanup) is not satisfied') -and
$bridgePlanCompletenessScriptText.Contains('Phase completion matrix contract violated:') -and
$bridgePlanCompletenessScriptText.Contains('missing non-empty title') -and
$bridgePlanCompletenessScriptText.Contains('missing checks') -and
$bridgePlanCompletenessScriptText.Contains('contains duplicate checks') -and
$bridgePlanCompletenessScriptText.Contains('missing required check:') -and
$bridgePlanCompletenessScriptText.Contains('has unexpected check:') -and
$bridgePlanCompletenessScriptText.Contains('missing satisfied field') -and
$bridgePlanCompletenessScriptText.Contains('satisfied mismatch with computed phase state') -and
$bridgePlanCompletenessScriptText.Contains('missing unsatisfied phase contract violation') -and
$bridgePlanCompletenessScriptText.Contains('missing non-empty reason') -and
$bridgePlanCompletenessScriptText.Contains('missing evidenceLocators') -and
$bridgePlanCompletenessScriptText.Contains('contains empty evidence locator') -and
$bridgePlanCompletenessScriptText.Contains('evidence locator format invalid') -and
$bridgePlanCompletenessScriptText.Contains('evidence locator prefix invalid') -and
$bridgePlanCompletenessScriptText.Contains('missing required locator label') -and
$bridgePlanCompletenessScriptText.Contains('phase0_active_dsp') -and
$bridgePlanCompletenessScriptText.Contains('phase1_policy_evaluations') -and
$bridgePlanCompletenessScriptText.Contains('phase2_within_target') -and
$bridgePlanCompletenessScriptText.Contains('phase3_direct_dependency') -and
$bridgePlanCompletenessScriptText.Contains('phase4_canary_violations') -and
$bridgePlanCompletenessScriptText.Contains('phase5_rollback_violations') -and
$bridgePlanCompletenessScriptText.Contains('phase6_cleanup_completed') -and
$bridgePlanCompletenessScriptText.Contains('Trigger audit evidence') -and
$bridgePlanCompletenessScriptText.Contains('Metric governance evidence') -and
$bridgePlanCompletenessScriptText.Contains('Flag dependency evidence') -and
$bridgePlanCompletenessScriptText.Contains('Enforcement adoption evidence') -and
$bridgePlanCompletenessScriptText.Contains('Enforcement source purity evidence') -and
$bridgePlanCompletenessScriptText.Contains('$($freshnessArtifact.Label) freshness breach') -and
$bridgePlanCompletenessScriptText.Contains('trigger_cleanup_completion_report.json') -and
$bridgePlanCompletenessScriptText.Contains('trigger_symbol_usage_report.json') -and
$bridgePlanCompletenessScriptText.Contains('observe_shim_usage_report.json') -and
$bridgePlanCompletenessScriptText.Contains('trigger_ast_report.json') -and
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
$bridgePlanCompletenessScriptText.Contains('triggerSymbolUsageIndex') -and
$bridgePlanCompletenessScriptText.Contains('observeShimUsageIndex') -and
$bridgePlanCompletenessScriptText.Contains('triggerAstIndex') -and
$bridgePlanCompletenessScriptText.Contains('triggerAuditIndex') -and
$bridgePlanCompletenessScriptText.Contains('cleanupPruneIndex') -and
$bridgePlanCompletenessScriptText.Contains('cleanupDeferredVerifyIndex') -and
$bridgePlanCompletenessScriptText.Contains('phase4DriftIndex') -and
$bridgePlanCompletenessScriptText.Contains('enforcementAdoptionIndex') -and
$bridgePlanCompletenessScriptText.Contains('enforcementSourcePurityIndex') -and
$bridgePlanCompletenessScriptText.Contains('rollbackMatrixIndex') -and
$bridgePlanCompletenessScriptText.Contains('facadeBypassIndex') -and
$bridgePlanCompletenessScriptText.Contains('canaryNormalizationIndex') -and
$bridgePlanCompletenessScriptText.Contains('metricGovernanceIndex') -and
$bridgePlanCompletenessScriptText.Contains('flagDependencyIndex') -and
$bridgePlanCompletenessScriptText.Contains('cleanupReadinessIndex') -and
$bridgePlanCompletenessScriptText.Contains('ownershipMigrationIndex') -and
$bridgePlanCompletenessScriptText.Contains('cleanupCompletionIndex') -and
$bridgePlanCompletenessScriptText.Contains('trigger-audit script token for authority transfer order validation') -and
$bridgePlanCompletenessScriptText.Contains('trigger audit gate must run before cleanup readiness gate') -and
$bridgePlanCompletenessScriptText.Contains('trigger audit gate must run before ownership migration gate') -and
$bridgePlanCompletenessScriptText.Contains('metric governance gate must run before bridge plan completeness gate') -and
$bridgePlanCompletenessScriptText.Contains('flag dependency graph gate must run before bridge plan completeness gate') -and
$bridgePlanCompletenessScriptText.Contains('Tier runner missing enforcement script tokens for phase order validation') -and
$bridgePlanCompletenessScriptText.Contains('enforcement adoption gate must run before cleanup readiness gate') -and
$bridgePlanCompletenessScriptText.Contains('enforcement source purity gate must run before cleanup readiness gate') -and
$bridgePlanCompletenessScriptText.Contains('Tier runner missing phase4/canary script tokens for order validation') -and
$bridgePlanCompletenessScriptText.Contains('phase4 generation drift gate must run before canary normalization gate') -and
$bridgePlanCompletenessScriptText.Contains('phase4 generation drift gate must run before bridge plan completeness gate') -and
$bridgePlanCompletenessScriptText.Contains('Tier runner missing rollback/facade/canary script tokens for order validation') -and
$bridgePlanCompletenessScriptText.Contains('rollback matrix gate must run before cleanup completion gate') -and
$bridgePlanCompletenessScriptText.Contains('facade bypass gate must run before cleanup completion gate') -and
$bridgePlanCompletenessScriptText.Contains('canary normalization gate must run before cleanup completion gate') -and
$bridgePlanCompletenessScriptText.Contains('Tier runner missing cleanup-prune/cleanup-deferred script tokens for order validation') -and
$bridgePlanCompletenessScriptText.Contains('cleanup prune gate must run before cleanup deferred verification gate') -and
$bridgePlanCompletenessScriptText.Contains('cleanup deferred verification gate must run before cleanup completion gate') -and
$bridgePlanCompletenessScriptText.Contains('readiness/ownership/cleanup-completion script tokens for order validation') -and
$bridgePlanCompletenessScriptText.Contains('cleanup readiness gate must run before cleanup completion gate') -and
$bridgePlanCompletenessScriptText.Contains('ownership migration gate must run before cleanup completion gate') -and
$bridgePlanCompletenessScriptText.Contains('cleanup completion gate must run before bridge plan completeness gate') -and
$bridgePlanCompletenessScriptText.Contains('executionOrderSequence') -and
$bridgePlanCompletenessScriptText.Contains('phaseEvidenceChronology') -and
$bridgePlanCompletenessScriptText.Contains('completionDefinitionStatus') -and
$bridgePlanCompletenessScriptText.Contains('Completion definition contract violated: bridgeRuntimeControllable is not satisfied') -and
$bridgePlanCompletenessScriptText.Contains('Completion definition contract violated: ciDeviationDetectable is not satisfied') -and
$bridgePlanCompletenessScriptText.Contains('Completion definition contract violated: rollbackSubsystemGranularity is not satisfied') -and
$bridgePlanCompletenessScriptText.Contains('Completion definition contract violated: deferredTriggerConvergence is not satisfied') -and
$bridgePlanCompletenessScriptText.Contains('Completion definition contract violated: noMajorIncidentIncrease is not satisfied') -and
$bridgePlanCompletenessScriptText.Contains('Completion definition matrix contract violated:') -and
$bridgePlanCompletenessScriptText.Contains('missing clause') -and
$bridgePlanCompletenessScriptText.Contains('missing non-empty reason') -and
$bridgePlanCompletenessScriptText.Contains('missing evidenceLocators') -and
$bridgePlanCompletenessScriptText.Contains('contains empty evidence locator') -and
$bridgePlanCompletenessScriptText.Contains('evidence locator format invalid') -and
$bridgePlanCompletenessScriptText.Contains('evidence locator prefix invalid') -and
$bridgePlanCompletenessScriptText.Contains('missing required locator label') -and
$bridgePlanCompletenessScriptText.Contains('completion_bridge_controllable_policy') -and
$bridgePlanCompletenessScriptText.Contains('completion_ci_detectable_tier_runner') -and
$bridgePlanCompletenessScriptText.Contains('completion_rollback_violations') -and
$bridgePlanCompletenessScriptText.Contains('completion_deferred_ready') -and
$bridgePlanCompletenessScriptText.Contains('completion_incident_canary_violations') -and
$bridgePlanCompletenessScriptText.Contains('backlogBeforeBridgeCompleteness') -and
$bridgePlanCompletenessScriptText.Contains('backlogResidualForwarding') -and
$bridgePlanCompletenessScriptText.Contains('enforceNoSpecFixedForwarded') -and
$bridgePlanCompletenessScriptText.Contains('isr-8_1-close-policy.json') -and
$bridgePlanCompletenessScriptText.Contains('isr_8_1_close_policy_v1') -and
$bridgePlanCompletenessScriptText.Contains('bridgePolicyPath') -and
$bridgePlanCompletenessScriptText.Contains('policyStatus') -and
$bridgePlanCompletenessScriptText.Contains('cleanupReferenceConsistency') -and
$bridgePlanCompletenessScriptText.Contains('deferredRegistryPath') -and
$bridgePlanCompletenessScriptText.Contains('sourceRoot') -and
$bridgePlanCompletenessScriptText.Contains('satisfied')
if (-not $bridgePlanCompletenessNeedsContracts) {
    throw 'Bridge plan completeness gate missing core contract checks'
}

$retirePressureScriptPath = Join-Path $repoRoot '.github\scripts\isr-verify-v73-retire-pressure-contract.ps1'
if (-not (Test-Path $retirePressureScriptPath)) {
    throw "Missing v7.3 retire pressure verifier script: $retirePressureScriptPath"
}

$retirePressureScriptText = Get-Content -LiteralPath $retirePressureScriptPath -Raw -Encoding UTF8
$retirePressureNeedsContracts =
$retirePressureScriptText.Contains('isr_v73_retire_pressure_report_v1') -and
$retirePressureScriptText.Contains('CI-RETIREPRESS-001') -and
$retirePressureScriptText.Contains('CI-RETIREPRESS-005') -and
$retirePressureScriptText.Contains('deferredDeleteFallbackQueue') -and
$retirePressureScriptText.Contains('setRetireBacklogCount') -and
$retirePressureScriptText.Contains('drainDeferredRetireQueues')
if (-not $retirePressureNeedsContracts) {
    throw 'v7.3 retire pressure verifier missing required contract checks'
}

$retireRtImmediateScriptPath = Join-Path $repoRoot '.github\scripts\isr-verify-v73-retire-rt-immediate-return.ps1'
if (-not (Test-Path $retireRtImmediateScriptPath)) {
    throw "Missing v7.3 retire RT immediate-return verifier script: $retireRtImmediateScriptPath"
}

$retireRtImmediateScriptText = Get-Content -LiteralPath $retireRtImmediateScriptPath -Raw -Encoding UTF8
$retireRtImmediateNeedsContracts =
$retireRtImmediateScriptText.Contains('isr_v73_retire_rt_immediate_return_report_v1') -and
$retireRtImmediateScriptText.Contains('CI-RETIRE-RT-001') -and
$retireRtImmediateScriptText.Contains('CI-RETIRE-RT-002') -and
$retireRtImmediateScriptText.Contains('CI-RETIRE-RT-003') -and
$retireRtImmediateScriptText.Contains('Set-StrictMode -Version Latest') -and
$retireRtImmediateScriptText.Contains('AudioEngine.Processing.AudioBlock.cpp') -and
$retireRtImmediateScriptText.Contains('AudioEngine.Processing.BlockDouble.cpp')
if (-not $retireRtImmediateNeedsContracts) {
    throw 'v7.3 retire RT immediate-return verifier missing required contract checks'
}

$prRequiredArtifactsScriptPath = Join-Path $repoRoot '.github\scripts\isr-verify-pr-required-artifacts.ps1'
if (-not (Test-Path $prRequiredArtifactsScriptPath)) {
    throw "Missing PR required artifacts verifier script: $prRequiredArtifactsScriptPath"
}

$prRequiredArtifactsScriptText = Get-Content -LiteralPath $prRequiredArtifactsScriptPath -Raw -Encoding UTF8
$prRequiredArtifactsNeedsContracts =
$prRequiredArtifactsScriptText.Contains('pr_required_artifacts_report_v1') -and
$prRequiredArtifactsScriptText.Contains('authority_inventory_report_v1') -and
$prRequiredArtifactsScriptText.Contains('authority_inventory_diff_report_v1') -and
$prRequiredArtifactsScriptText.Contains('safety_regression_report_v1') -and
$prRequiredArtifactsScriptText.Contains('pr_sla_report_v1') -and
$prRequiredArtifactsScriptText.Contains('validator_tiering_report_v3') -and
$prRequiredArtifactsScriptText.Contains('documentation_scope_rule_report_v1') -and
$prRequiredArtifactsScriptText.Contains('design_docs_coverage_report_v1') -and
$prRequiredArtifactsScriptText.Contains('runtime_coordinator_state_machine_report_v1') -and
$prRequiredArtifactsScriptText.Contains('taxonomy_phase_mapping_report_v1') -and
$prRequiredArtifactsScriptText.Contains('Practical_Stable_ISR_Runtime_topology_diff_*.md') -and
$prRequiredArtifactsScriptText.Contains('documentation_scope_rule_report must be ready=true') -and
$prRequiredArtifactsScriptText.Contains('design_docs_coverage_report must be ready=true') -and
$prRequiredArtifactsScriptText.Contains('documentationScopeReady') -and
$prRequiredArtifactsScriptText.Contains('designDocsCoverageReady') -and
$prRequiredArtifactsScriptText.Contains('runtimeCoordinatorStateMachineReady') -and
$prRequiredArtifactsScriptText.Contains('taxonomyPhaseMappingReady') -and
$prRequiredArtifactsScriptText.Contains('releaseRequiresExhaustive') -and
$prRequiredArtifactsScriptText.Contains('releaseWindow') -and
$prRequiredArtifactsScriptText.Contains('openedAtSource') -and
$prRequiredArtifactsScriptText.Contains('eventHeadSha') -and
$prRequiredArtifactsScriptText.Contains('currentHeadSha') -and
$prRequiredArtifactsScriptText.Contains('staleEvaluation') -and
$prRequiredArtifactsScriptText.Contains('needsRevalidation=true requires labelSuggestions to include needs-revalidation')
if (-not $prRequiredArtifactsNeedsContracts) {
    throw 'PR required artifacts verifier missing required contract checks'
}

$taxonomyPhaseMappingScriptPath = Join-Path $repoRoot '.github\scripts\isr-verify-taxonomy-phase-mapping.ps1'
if (-not (Test-Path $taxonomyPhaseMappingScriptPath)) {
    throw "Missing taxonomy phase mapping verifier script: $taxonomyPhaseMappingScriptPath"
}

$taxonomyPhaseMappingScriptText = Get-Content -LiteralPath $taxonomyPhaseMappingScriptPath -Raw -Encoding UTF8
$taxonomyPhaseMappingNeedsContracts =
$taxonomyPhaseMappingScriptText.Contains('taxonomy_phase_mapping_report.json') -and
$taxonomyPhaseMappingScriptText.Contains('taxonomy_phase_mapping_report_v1') -and
$taxonomyPhaseMappingScriptText.Contains('Class-A/B/C') -and
$taxonomyPhaseMappingScriptText.Contains('Class-D/E') -and
$taxonomyPhaseMappingScriptText.Contains('Class-F')
if (-not $taxonomyPhaseMappingNeedsContracts) {
    throw 'Taxonomy phase mapping verifier missing required contract checks'
}

$designDocsCoverageScriptPath = Join-Path $repoRoot '.github\scripts\isr-verify-design-docs-coverage.ps1'
if (-not (Test-Path $designDocsCoverageScriptPath)) {
    throw "Missing design docs coverage verifier script: $designDocsCoverageScriptPath"
}

$designDocsCoverageScriptText = Get-Content -LiteralPath $designDocsCoverageScriptPath -Raw -Encoding UTF8
$designDocsCoverageNeedsContracts =
$designDocsCoverageScriptText.Contains('design_docs_coverage_report_v1') -and
$designDocsCoverageScriptText.Contains('Practical_Stable_ISR_Runtime_基本計画書_v3_1.md') -and
$designDocsCoverageScriptText.Contains('ISR_Runtime_実装統治規約_v1_1.md') -and
$designDocsCoverageScriptText.Contains('Practical_Stable_ISR_Runtime_詳細設計_v1_2.md') -and
$designDocsCoverageScriptText.Contains('Practical_Stable_ISR_Runtime_フェーズ別実装タスク分解_v1_0.md') -and
$designDocsCoverageScriptText.Contains('Design document token missing:')
if (-not $designDocsCoverageNeedsContracts) {
    throw 'Design docs coverage verifier missing required contract checks'
}

$prSlaScriptPath = Join-Path $repoRoot '.github\scripts\isr-verify-pr-sla.ps1'
if (-not (Test-Path $prSlaScriptPath)) {
    throw "Missing PR SLA verifier script: $prSlaScriptPath"
}

$prSlaScriptText = Get-Content -LiteralPath $prSlaScriptPath -Raw -Encoding UTF8
$prSlaNeedsReleaseContracts =
$prSlaScriptText.Contains('releaseRequiresExhaustive') -and
$prSlaScriptText.Contains('Release window requires exhaustive tier for all classes') -and
$prSlaScriptText.Contains('releaseWindow') -and
$prSlaScriptText.Contains('gitRef') -and
$prSlaScriptText.Contains('authority_inventory_report.json') -and
$prSlaScriptText.Contains('inventory_diff_report.json') -and
$prSlaScriptText.Contains('Required note check failed: note=soak medium 24h') -and
$prSlaScriptText.Contains('Required note check failed: note=soak long 72h') -and
$prSlaScriptText.Contains('Required note check failed: note=soak extreme 1week') -and
$prSlaScriptText.Contains('Required note check failed: note=break-glass approval') -and
$prSlaScriptText.Contains('Required note check failed: note=rollback plan required') -and
$prSlaScriptText.Contains('Required note check failed: note=runtime code change zero') -and
$prSlaScriptText.Contains('PR class policy missing requiredNotes:') -and
$prSlaScriptText.Contains('Policy boolean check failed: runtimeCodeChangeZeroRequired=true') -and
$prSlaScriptText.Contains('Policy boolean check failed: inventoryDiffStructuralInvariantRequired=true') -and
$prSlaScriptText.Contains('runtimeCodeChangeZeroConfidence') -and
$prSlaScriptText.Contains('Resolve-RuntimeCodeChangeSignal') -and
$prSlaScriptText.Contains('GITHUB_EVENT_PATH') -and
$prSlaScriptText.Contains('SoakMinutes must be non-negative') -and
$prSlaScriptText.Contains('requiredNotes') -and
$prSlaScriptText.Contains('soakMinutes') -and
$prSlaScriptText.Contains('breakglassReportReady') -and
$prSlaScriptText.Contains('RequireDeclaredClass') -and
$prSlaScriptText.Contains('Resolve-DeclaredClassFromEventLabels') -and
$prSlaScriptText.Contains('Declared PR class is required but missing') -and
$prSlaScriptText.Contains('declaredClassSource') -and
$prSlaScriptText.Contains('Resolve-OpenedAtFromEvent') -and
$prSlaScriptText.Contains('openedAtSource') -and
$prSlaScriptText.Contains('Resolve-EventHeadSha') -and
$prSlaScriptText.Contains('PR SLA evaluation is stale') -and
$prSlaScriptText.Contains('eventHeadSha') -and
$prSlaScriptText.Contains('currentHeadSha') -and
$prSlaScriptText.Contains('staleEvaluation')
if (-not $prSlaNeedsReleaseContracts) {
    throw 'PR SLA verifier missing release-window and required-note contracts'
}

if (-not ($tierRunnerText.Contains('$env:GITHUB_EVENT_NAME -eq ''pull_request''') -and
        $tierRunnerText.Contains('$prSlaArgs[''RequireDeclaredClass''] = $true'))) {
    throw 'Tier runner missing pull_request declared-class enforcement wiring for PR SLA verifier'
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
