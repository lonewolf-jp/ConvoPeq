$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$planPath = Join-Path $repoRoot 'doc\work\bridge_runtime_migration_plan.md'
$bridgePolicyPath = Join-Path $repoRoot 'doc\work\ISR_Bridge_Runtime_AI_暴走防止規約.md'
$tierRunnerPath = Join-Path $repoRoot '.github\scripts\isr-run-tiered-verification.ps1'
$evidenceDir = Join-Path $repoRoot 'evidence'
$reportPath = Join-Path $evidenceDir 'bridge_plan_completeness_report.json'

foreach ($path in @($planPath, $bridgePolicyPath, $tierRunnerPath)) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Missing required file: $path"
    }
}

if (-not (Test-Path -LiteralPath $evidenceDir)) {
    New-Item -Path $evidenceDir -ItemType Directory | Out-Null
}

$planText = Get-Content -LiteralPath $planPath -Raw -Encoding UTF8
$bridgePolicyText = Get-Content -LiteralPath $bridgePolicyPath -Raw -Encoding UTF8
$tierRunnerText = Get-Content -LiteralPath $tierRunnerPath -Raw -Encoding UTF8
$violations = New-Object System.Collections.Generic.List[string]
$artifactFreshnessWindowMinutes = 1440
$triggerAuditArtifactPath = Join-Path $evidenceDir 'trigger_audit_report.json'
$triggerCleanupReadinessArtifactPath = Join-Path $evidenceDir 'trigger_cleanup_readiness_report.json'
$triggerCleanupCompletionArtifactPath = Join-Path $evidenceDir 'trigger_cleanup_completion_report.json'
$ownershipMigrationArtifactPath = Join-Path $evidenceDir 'ownership_migration_report.json'
$backlogResidualArtifactPath = Join-Path $evidenceDir 'backlog_specfixed_residual_report.json'
$canaryNormalizationArtifactPath = Join-Path $evidenceDir 'canary_baseline_normalization_report.json'
$metricGovernanceArtifactPath = Join-Path $evidenceDir 'metric_governance_report.json'
$flagDependencyArtifactPath = Join-Path $evidenceDir 'flag_dependency_graph_report.json'
$enforcementAdoptionArtifactPath = Join-Path $evidenceDir 'enforcement_adoption_report.json'
$enforcementSourcePurityArtifactPath = Join-Path $evidenceDir 'enforcement_source_purity_report.json'

foreach ($phase in @(
        'Phase 0: 統制基盤の先行導入',
        'Phase 1: 計測可能トリガー化',
        'Phase 2: enforcement 高度化（grep→AST）',
        'Phase 3: facade 統制',
        'Phase 4: crossfade 専用移行',
        'Phase 5: rollback hierarchy 導入',
        'Phase 6: cleanup（trigger達成後）'
    )) {
    if ($planText -notmatch [regex]::Escape($phase)) {
        $violations.Add("Plan missing phase section: $phase")
    }
}

foreach ($section in @(
        '## 4. Invariant（最小セット）',
        '## 7. CI / 検証運用',
        '## 7.1 Validator tiering',
        '## 7.2 PR canary metrics',
        '## 7.3 CI rule self-test',
        '## 8. メトリクス運用規約（Metric Governance）',
        '## 9. Trigger 一覧（機械判定）',
        '## 10. allowlist 運用ポリシー',
        '## 11. 既知リスクと抑止',
        '## 12. この計画での「完成」定義'
    )) {
    if ($planText -notmatch [regex]::Escape($section)) {
        $violations.Add("Plan missing governance section: $section")
    }
}

foreach ($slaClause in @(
        '### exhaustive fail SLA（必須）',
        'HB violation: 24h',
        'payload mismatch: 72h',
        'baseline window normalization を実施'
    )) {
    if ($planText -notmatch [regex]::Escape($slaClause)) {
        $violations.Add("Plan missing SLA/canary clause: $slaClause")
    }
}

foreach ($canaryMetricClause in @(
        'XRUN delta',
        'callback jitter',
        'retire latency',
        'crossfade peak'
    )) {
    if ($planText -notmatch [regex]::Escape($canaryMetricClause)) {
        $violations.Add("Plan missing canary metric clause: $canaryMetricClause")
    }
}

foreach ($knownRisk in @(
        'metrics explosion',
        'flag interaction hell',
        'RTAuxMutable 肥大化',
        'validator decay',
        'rollback 逆依存崩壊'
    )) {
    if ($planText -notmatch [regex]::Escape($knownRisk)) {
        $violations.Add("Plan missing known-risk clause: $knownRisk")
    }
}

foreach ($completionClause in @(
        'bridge runtime が制御不能化しない',
        'CIで逸脱を機械検知できる',
        'rollback が subsystem 粒度で機能する',
        'deferred が trigger 駆動で確実に縮退する',
        '主要事故（UAF/RT blocking/partial visibility/stale access）が実運用で増加しない'
    )) {
    if ($planText -notmatch [regex]::Escape($completionClause)) {
        $violations.Add("Plan missing completion clause: $completionClause")
    }
}

foreach ($planAntiPurityClause in @(
        '完全ISRへの直行ではなく',
        'bridge runtime の事故率を継続的に下げる統制運用'
    )) {
    if ($planText -notmatch [regex]::Escape($planAntiPurityClause)) {
        $violations.Add("Plan missing anti-purity clause: $planAntiPurityClause")
    }
}

foreach ($policyGuardrail in @(
        '### 1.1 1PR = 1責務',
        '### 1.2 “cleanup” 単独PR禁止',
        '### 3.1 callback observe固定を最優先',
        '### 4.2 crossfade rewrite 禁止',
        '### 6.2 validator 削除禁止',
        '### 7.3 CI rule self-test 必須',
        '### 9.2 subsystem rollback 維持',
        '- 全面rewrite'
    )) {
    if ($bridgePolicyText -notmatch [regex]::Escape($policyGuardrail)) {
        $violations.Add("Policy missing guardrail clause: $policyGuardrail")
    }
}

foreach ($reviewPriorityClause in @(
        '1. XRUN悪化がないか',
        '2. click/pop悪化がないか',
        '3. rollback可能か',
        '4. dual authority暴走していないか',
        '5. observe path増殖していないか',
        '6. RT-visible mutation増えていないか',
        '7. cleanup先走りしていないか',
        '8. purity追求になっていないか'
    )) {
    if ($bridgePolicyText -notmatch [regex]::Escape($reviewPriorityClause)) {
        $violations.Add("Policy missing review-priority clause: $reviewPriorityClause")
    }
}

foreach ($policyAntiPurityClause in @(
        '「理想ISRを作る」',
        '「bridge runtime を長期間崩壊させない統制システム」'
    )) {
    if ($bridgePolicyText -notmatch [regex]::Escape($policyAntiPurityClause)) {
        $violations.Add("Policy missing anti-purity clause: $policyAntiPurityClause")
    }
}

foreach ($prohibitedAiActionClause in @(
        '- 全面rewrite',
        '- global mutable purge',
        '- full ISR rewrite',
        '- crossfade全面刷新',
        '- RuntimeGraph肥大化',
        '- validator削除',
        '- metric大量追加',
        '- flag依存複雑化',
        '- facade多層化',
        '- template abstraction 増殖',
        '- “generic runtime framework” 化'
    )) {
    if ($bridgePolicyText -notmatch [regex]::Escape($prohibitedAiActionClause)) {
        $violations.Add("Policy missing prohibited-ai-action clause: $prohibitedAiActionClause")
    }
}

foreach ($invariant in @('IR-A', 'IR-B', 'IR-C', 'IR-D', 'IR-E', 'IR-F', 'IR-G')) {
    if ($planText -notmatch [regex]::Escape($invariant)) {
        $violations.Add("Plan missing invariant label: $invariant")
    }
}

$executionOrderItems = @(
    'ObserveToken formalization（極小）',
    'RTLocalState + RTAuxMutable 分離',
    '新規legacy参照禁止（CI）',
    'callback中snapshot固定 + IR-E',
    'grep/lint enforcement',
    'clang-tidy/symbol enforcement',
    'retire facade + bypass CI + execution metrics',
    'crossfade専用移行（latency分離含む）',
    'rollback hierarchy + matrix',
    'validator tiering + SLA',
    'trigger達成項目のcleanup'
)

$executionOrderSequence = New-Object System.Collections.Generic.List[object]
$lastExecutionOrderIndex = -1
$executionOrderIsSorted = $true

foreach ($executionOrderItem in $executionOrderItems) {
    if ($planText -notmatch [regex]::Escape($executionOrderItem)) {
        $violations.Add("Plan missing execution-order item: $executionOrderItem")
        $executionOrderSequence.Add([ordered]@{
                item = $executionOrderItem
                index = -1
            }) | Out-Null
        continue
    }

    $currentIndex = $planText.IndexOf($executionOrderItem)
    $executionOrderSequence.Add([ordered]@{
            item = $executionOrderItem
            index = $currentIndex
        }) | Out-Null

    if ($currentIndex -lt $lastExecutionOrderIndex) {
        $executionOrderIsSorted = $false
    }

    $lastExecutionOrderIndex = $currentIndex
}

if (-not $executionOrderIsSorted) {
    $violations.Add('Plan execution-order sequence violated: section 13 items are out of order')
}

$requiredScripts = @(
    '.github/scripts/isr-verify-trigger-symbol-usage.ps1',
    '.github/scripts/isr-verify-observe-shim-usage.ps1',
    '.github/scripts/isr-verify-trigger-ast.ps1',
    '.github/scripts/isr-trigger-audit.ps1',
    '.github/scripts/isr-prune-cleanup-deferred.ps1',
    '.github/scripts/isr-verify-cleanup-deferred.ps1',
    '.github/scripts/isr-verify-validator-tiering.ps1',
    '.github/scripts/isr-verify-policy-top-level-governance.ps1',
    '.github/scripts/isr-verify-workflow-dispatch-input-policy.ps1',
    '.github/scripts/isr-verify-8_1-close-policy.ps1',
    '.github/scripts/isr-verify-8_1-workflow-input-contract.ps1',
    '.github/scripts/isr-verify-8_1-workflow-input-coherence.ps1',
    '.github/scripts/isr-verify-trigger-ast.ps1',
    '.github/scripts/isr-verify-facade-bypass.ps1',
    '.github/scripts/isr-verify-crossfade-observable-state.ps1',
    '.github/scripts/isr-verify-canary-baseline-normalization.ps1',
    '.github/scripts/isr-verify-phase4-generation-drift.ps1',
    '.github/scripts/isr-verify-metric-governance.ps1',
    '.github/scripts/isr-verify-flag-dependency-graph.ps1',
    '.github/scripts/isr-verify-rollback-matrix.ps1',
    '.github/scripts/isr-verify-trigger-cleanup-readiness.ps1',
    '.github/scripts/isr-verify-ownership-migration.ps1',
    '.github/scripts/isr-verify-trigger-cleanup-completion.ps1',
    '.github/scripts/isr-verify-backlog-specfixed-residual.ps1'
)

foreach ($scriptPath in $requiredScripts) {
    if ($tierRunnerText -notmatch [regex]::Escape("'$scriptPath'")) {
        $violations.Add("Tier runner missing plan-completeness script wiring: $scriptPath")
    }
}

$backlogScriptToken = "'.github/scripts/isr-verify-backlog-specfixed-residual.ps1'"
$triggerSymbolUsageScriptToken = "'.github/scripts/isr-verify-trigger-symbol-usage.ps1'"
$observeShimUsageScriptToken = "'.github/scripts/isr-verify-observe-shim-usage.ps1'"
$triggerAstScriptToken = "'.github/scripts/isr-verify-trigger-ast.ps1'"
$triggerAuditScriptToken = "'.github/scripts/isr-trigger-audit.ps1'"
$metricGovernanceScriptToken = "'.github/scripts/isr-verify-metric-governance.ps1'"
$flagDependencyScriptToken = "'.github/scripts/isr-verify-flag-dependency-graph.ps1'"
$cleanupReadinessScriptToken = "'.github/scripts/isr-verify-trigger-cleanup-readiness.ps1'"
$cleanupPruneScriptToken = "'.github/scripts/isr-prune-cleanup-deferred.ps1'"
$cleanupDeferredVerifyScriptToken = "'.github/scripts/isr-verify-cleanup-deferred.ps1'"
$enforcementAdoptionScriptToken = "'.github/scripts/isr-verify-enforcement-adoption.ps1'"
$enforcementSourcePurityScriptToken = "'.github/scripts/isr-verify-enforcement-source-purity.ps1'"
$phase4DriftScriptToken = "'.github/scripts/isr-verify-phase4-generation-drift.ps1'"
$rollbackMatrixScriptToken = "'.github/scripts/isr-verify-rollback-matrix.ps1'"
$facadeBypassScriptToken = "'.github/scripts/isr-verify-facade-bypass.ps1'"
$canaryNormalizationScriptToken = "'.github/scripts/isr-verify-canary-baseline-normalization.ps1'"
$ownershipMigrationScriptToken = "'.github/scripts/isr-verify-ownership-migration.ps1'"
$cleanupCompletionScriptToken = "'.github/scripts/isr-verify-trigger-cleanup-completion.ps1'"
$bridgeScriptToken = "'.github/scripts/isr-verify-bridge-plan-completeness.ps1'"
$backlogScriptIndex = $tierRunnerText.IndexOf($backlogScriptToken)
$triggerSymbolUsageScriptIndex = $tierRunnerText.IndexOf($triggerSymbolUsageScriptToken)
$observeShimUsageScriptIndex = $tierRunnerText.IndexOf($observeShimUsageScriptToken)
$triggerAstScriptIndex = $tierRunnerText.IndexOf($triggerAstScriptToken)
$triggerAuditScriptIndex = $tierRunnerText.IndexOf($triggerAuditScriptToken)
$metricGovernanceScriptIndex = $tierRunnerText.IndexOf($metricGovernanceScriptToken)
$flagDependencyScriptIndex = $tierRunnerText.IndexOf($flagDependencyScriptToken)
$cleanupReadinessScriptIndex = $tierRunnerText.IndexOf($cleanupReadinessScriptToken)
$cleanupPruneScriptIndex = $tierRunnerText.IndexOf($cleanupPruneScriptToken)
$cleanupDeferredVerifyScriptIndex = $tierRunnerText.IndexOf($cleanupDeferredVerifyScriptToken)
$enforcementAdoptionScriptIndex = $tierRunnerText.IndexOf($enforcementAdoptionScriptToken)
$enforcementSourcePurityScriptIndex = $tierRunnerText.IndexOf($enforcementSourcePurityScriptToken)
$phase4DriftScriptIndex = $tierRunnerText.IndexOf($phase4DriftScriptToken)
$rollbackMatrixScriptIndex = $tierRunnerText.IndexOf($rollbackMatrixScriptToken)
$facadeBypassScriptIndex = $tierRunnerText.IndexOf($facadeBypassScriptToken)
$canaryNormalizationScriptIndex = $tierRunnerText.IndexOf($canaryNormalizationScriptToken)
$ownershipMigrationScriptIndex = $tierRunnerText.IndexOf($ownershipMigrationScriptToken)
$cleanupCompletionScriptIndex = $tierRunnerText.IndexOf($cleanupCompletionScriptToken)
$bridgeScriptIndex = $tierRunnerText.IndexOf($bridgeScriptToken)
$scriptOrderSatisfied = $false
$expectedBacklogPath = Join-Path $repoRoot 'doc\work\ISR_Completeness_Risk_Backlog.md'
$expectedDeferredRegistryPath = Join-Path $repoRoot '.github\isr-cleanup-deferred.json'
$expectedSourceRoot = Join-Path $repoRoot 'src'
$actualBacklogPath = $null
$actualDeferredRegistryPath = $null
$actualSourceRoot = $null
$backlogPathSatisfied = $false
$deferredRegistryPathSatisfied = $false
$sourceRootSatisfied = $false

if ($backlogScriptIndex -lt 0 -or $bridgeScriptIndex -lt 0) {
    $violations.Add('Tier runner missing backlog/bridge completeness script tokens for order validation')
}
elseif ($backlogScriptIndex -ge $bridgeScriptIndex) {
    $violations.Add('Tier runner script order contract violated: backlog spec-fixed residual gate must run before bridge plan completeness gate')
}
else {
    $scriptOrderSatisfied = $true
}

if ($triggerSymbolUsageScriptIndex -lt 0 -or $observeShimUsageScriptIndex -lt 0 -or $triggerAstScriptIndex -lt 0 -or $triggerAuditScriptIndex -lt 0) {
    $violations.Add('Tier runner missing trigger preflight script tokens for order validation')
}
else {
    if ($triggerSymbolUsageScriptIndex -ge $triggerAuditScriptIndex) {
        $violations.Add('Tier runner script order contract violated: trigger symbol usage gate must run before trigger audit gate')
    }
    if ($observeShimUsageScriptIndex -ge $triggerAuditScriptIndex) {
        $violations.Add('Tier runner script order contract violated: observe shim usage gate must run before trigger audit gate')
    }
    if ($triggerAstScriptIndex -ge $triggerAuditScriptIndex) {
        $violations.Add('Tier runner script order contract violated: trigger AST gate must run before trigger audit gate')
    }
}

if ($metricGovernanceScriptIndex -lt 0 -or $flagDependencyScriptIndex -lt 0) {
    $violations.Add('Tier runner missing metric-governance/flag-dependency script tokens for order validation')
}
else {
    if ($metricGovernanceScriptIndex -ge $bridgeScriptIndex) {
        $violations.Add('Tier runner script order contract violated: metric governance gate must run before bridge plan completeness gate')
    }
    if ($flagDependencyScriptIndex -ge $bridgeScriptIndex) {
        $violations.Add('Tier runner script order contract violated: flag dependency graph gate must run before bridge plan completeness gate')
    }
}

if ($enforcementAdoptionScriptIndex -lt 0 -or $enforcementSourcePurityScriptIndex -lt 0) {
    $violations.Add('Tier runner missing enforcement script tokens for phase order validation')
}
else {
    if ($enforcementAdoptionScriptIndex -ge $cleanupReadinessScriptIndex) {
        $violations.Add('Tier runner script order contract violated: enforcement adoption gate must run before cleanup readiness gate')
    }
    if ($enforcementSourcePurityScriptIndex -ge $cleanupReadinessScriptIndex) {
        $violations.Add('Tier runner script order contract violated: enforcement source purity gate must run before cleanup readiness gate')
    }
}

if ($cleanupPruneScriptIndex -lt 0 -or $cleanupDeferredVerifyScriptIndex -lt 0) {
    $violations.Add('Tier runner missing cleanup-prune/cleanup-deferred script tokens for order validation')
}
else {
    if ($cleanupPruneScriptIndex -ge $cleanupDeferredVerifyScriptIndex) {
        $violations.Add('Tier runner script order contract violated: cleanup prune gate must run before cleanup deferred verification gate')
    }
    if ($cleanupDeferredVerifyScriptIndex -ge $cleanupCompletionScriptIndex) {
        $violations.Add('Tier runner script order contract violated: cleanup deferred verification gate must run before cleanup completion gate')
    }
}

if ($phase4DriftScriptIndex -lt 0 -or $canaryNormalizationScriptIndex -lt 0) {
    $violations.Add('Tier runner missing phase4/canary script tokens for order validation')
}
else {
    if ($phase4DriftScriptIndex -ge $canaryNormalizationScriptIndex) {
        $violations.Add('Tier runner script order contract violated: phase4 generation drift gate must run before canary normalization gate')
    }
    if ($phase4DriftScriptIndex -ge $bridgeScriptIndex) {
        $violations.Add('Tier runner script order contract violated: phase4 generation drift gate must run before bridge plan completeness gate')
    }
}

if ($rollbackMatrixScriptIndex -lt 0 -or $facadeBypassScriptIndex -lt 0 -or $canaryNormalizationScriptIndex -lt 0) {
    $violations.Add('Tier runner missing rollback/facade/canary script tokens for order validation')
}
else {
    if ($rollbackMatrixScriptIndex -ge $cleanupCompletionScriptIndex) {
        $violations.Add('Tier runner script order contract violated: rollback matrix gate must run before cleanup completion gate')
    }
    if ($facadeBypassScriptIndex -ge $cleanupCompletionScriptIndex) {
        $violations.Add('Tier runner script order contract violated: facade bypass gate must run before cleanup completion gate')
    }
    if ($canaryNormalizationScriptIndex -ge $cleanupCompletionScriptIndex) {
        $violations.Add('Tier runner script order contract violated: canary normalization gate must run before cleanup completion gate')
    }
}

if ($cleanupReadinessScriptIndex -lt 0 -or $ownershipMigrationScriptIndex -lt 0 -or $cleanupCompletionScriptIndex -lt 0) {
    $violations.Add('Tier runner missing readiness/ownership/cleanup-completion script tokens for order validation')
}
else {
    if ($triggerAuditScriptIndex -lt 0) {
        $violations.Add('Tier runner missing trigger-audit script token for authority transfer order validation')
    }
    elseif ($triggerAuditScriptIndex -ge $cleanupReadinessScriptIndex) {
        $violations.Add('Tier runner script order contract violated: trigger audit gate must run before cleanup readiness gate')
    }

    if ($triggerAuditScriptIndex -lt 0) {
        $violations.Add('Tier runner missing trigger-audit script token for ownership migration order validation')
    }
    elseif ($triggerAuditScriptIndex -ge $ownershipMigrationScriptIndex) {
        $violations.Add('Tier runner script order contract violated: trigger audit gate must run before ownership migration gate')
    }

    if ($cleanupReadinessScriptIndex -ge $cleanupCompletionScriptIndex) {
        $violations.Add('Tier runner script order contract violated: cleanup readiness gate must run before cleanup completion gate')
    }
    if ($ownershipMigrationScriptIndex -ge $cleanupCompletionScriptIndex) {
        $violations.Add('Tier runner script order contract violated: ownership migration gate must run before cleanup completion gate')
    }
    if ($cleanupCompletionScriptIndex -ge $bridgeScriptIndex) {
        $violations.Add('Tier runner script order contract violated: cleanup completion gate must run before bridge plan completeness gate')
    }
}

$backlogEnforceForwardAnchor = "elseif (`$scriptPath -eq '.github/scripts/isr-verify-backlog-specfixed-residual.ps1')"
$backlogEnforceForwardCall = '& $scriptPath -EnforceNoSpecFixed'
$backlogEnforceForwardSatisfied = $tierRunnerText.Contains($backlogEnforceForwardAnchor) -and $tierRunnerText.Contains($backlogEnforceForwardCall)
if (-not $backlogEnforceForwardSatisfied) {
    $violations.Add('Tier runner missing EnforceNoSpecFixed forwarding for backlog spec-fixed residual gate')
}

$requiredArtifacts = @(
    @{ Path = (Join-Path $evidenceDir 'trigger_symbol_usage_report.json'); Schema = 'trigger_symbol_usage_report_v1' },
    @{ Path = (Join-Path $evidenceDir 'observe_shim_usage_report.json'); Schema = 'observe_shim_usage_report_v1' },
    @{ Path = (Join-Path $evidenceDir 'trigger_ast_report.json'); Schema = 'trigger_ast_report_v1' },
    @{ Path = (Join-Path $evidenceDir 'trigger_audit_report.json'); Schema = 'trigger_audit_report_v1' },
    @{ Path = (Join-Path $evidenceDir 'validator_tiering_report.json'); Schema = 'validator_tiering_report_v3' },
    @{ Path = (Join-Path $evidenceDir 'policy_top_level_governance_report.json'); Schema = 'policy_top_level_governance_report_v2' },
    @{ Path = (Join-Path $evidenceDir 'workflow_dispatch_input_policy_report.json'); Schema = 'workflow_dispatch_input_policy_report_v1' },
    @{ Path = (Join-Path $evidenceDir 'close_policy_8_1_report.json'); Schema = 'close_policy_8_1_report_v1' },
    @{ Path = (Join-Path $evidenceDir 'close_policy_8_1_workflow_input_contract_report.json'); Schema = 'close_policy_8_1_workflow_input_contract_report_v1' },
    @{ Path = (Join-Path $evidenceDir 'close_policy_8_1_workflow_input_coherence_report.json'); Schema = 'close_policy_8_1_workflow_input_coherence_report_v1' },
    @{ Path = (Join-Path $evidenceDir 'facade_bypass_report.json'); Schema = 'facade_bypass_report_v1' },
    @{ Path = (Join-Path $evidenceDir 'crossfade_observable_state_report.json'); Schema = 'crossfade_observable_state_report_v1' },
    @{ Path = (Join-Path $evidenceDir 'canary_baseline_normalization_report.json'); Schema = 'canary_baseline_normalization_report_v1' },
    @{ Path = (Join-Path $evidenceDir 'phase4_generation_drift_report.json'); Schema = 'phase4_generation_drift_report_v1' },
    @{ Path = (Join-Path $evidenceDir 'enforcement_adoption_report.json'); Schema = 'enforcement_adoption_report_v1' },
    @{ Path = (Join-Path $evidenceDir 'enforcement_source_purity_report.json'); Schema = 'enforcement_source_purity_report_v1' },
    @{ Path = (Join-Path $evidenceDir 'metric_governance_report.json'); Schema = 'metric_governance_report_v2' },
    @{ Path = (Join-Path $evidenceDir 'flag_dependency_graph_report.json'); Schema = 'flag_dependency_graph_report_v1' },
    @{ Path = (Join-Path $evidenceDir 'rollback_compatibility_report.json'); Schema = 'rollback_compatibility_report_v1' },
    @{ Path = (Join-Path $evidenceDir 'trigger_cleanup_readiness_report.json'); Schema = 'trigger_cleanup_readiness_report_v1' },
    @{ Path = (Join-Path $evidenceDir 'cleanup_deferred_prune_report.json'); Schema = 'cleanup_deferred_prune_report_v1' },
    @{ Path = (Join-Path $evidenceDir 'cleanup_deferred_report.json'); Schema = 'cleanup_deferred_report_v1' },
    @{ Path = (Join-Path $evidenceDir 'ownership_migration_report.json'); Schema = 'ownership_migration_report_v2' },
    @{ Path = (Join-Path $evidenceDir 'trigger_cleanup_completion_report.json'); Schema = 'trigger_cleanup_completion_report_v1' },
    @{ Path = (Join-Path $evidenceDir 'backlog_specfixed_residual_report.json'); Schema = 'backlog_specfixed_residual_report_v1' }
)

$requiredPolicies = @(
    @{ Path = (Join-Path $repoRoot '.github\isr-8_1-close-policy.json'); Schema = 'isr_8_1_close_policy_v1' },
    @{ Path = (Join-Path $repoRoot '.github\isr-validator-tiering-policy.json'); Schema = 'isr_validator_tiering_policy_v1' },
    @{ Path = (Join-Path $repoRoot '.github\isr-workflow-dispatch-input-policy.json'); Schema = 'isr_workflow_dispatch_input_policy_v1' }
)

$requiredAllowlists = @(
    @{ Path = (Join-Path $repoRoot '.github\isr-trigger-symbol-allowlist.json'); Schema = 'trigger_symbol_allowlist_v1' },
    @{ Path = (Join-Path $repoRoot '.github\isr-observe-shim-allowlist.json'); Schema = 'observe_shim_allowlist_v1' },
    @{ Path = (Join-Path $repoRoot '.github\isr-cleanup-deferred.json'); Schema = 'cleanup_deferred_registry_v1' }
)

$artifactStatus = New-Object System.Collections.Generic.List[object]
foreach ($artifact in $requiredArtifacts) {
    $exists = Test-Path -LiteralPath $artifact.Path
    $actualSchema = $null

    if ($exists) {
        try {
            $content = Get-Content -LiteralPath $artifact.Path -Raw -Encoding UTF8 | ConvertFrom-Json
            $actualSchema = "$($content.schema)"
            if ($actualSchema -ne "$($artifact.Schema)") {
                $violations.Add("Artifact schema mismatch: path=$($artifact.Path) expected=$($artifact.Schema) actual=$actualSchema")
            }
        }
        catch {
            $violations.Add("Artifact parse failed: path=$($artifact.Path) reason=$($_.Exception.Message)")
        }
    }
    else {
        $violations.Add("Missing required plan evidence artifact: $($artifact.Path)")
    }

    $artifactStatus.Add([ordered]@{
            path           = $artifact.Path
            expectedSchema = $artifact.Schema
            exists         = $exists
            actualSchema   = $actualSchema
        }) | Out-Null
}

$cleanupDeferredPruneArtifactPath = Join-Path $evidenceDir 'cleanup_deferred_prune_report.json'
if (Test-Path -LiteralPath $cleanupDeferredPruneArtifactPath) {
    try {
        $cleanupDeferredPrune = Get-Content -LiteralPath $cleanupDeferredPruneArtifactPath -Raw -Encoding UTF8 | ConvertFrom-Json

        if ($null -eq $cleanupDeferredPrune.violations) {
            $violations.Add('Cleanup prune evidence missing violations field')
        }
        elseif (@($cleanupDeferredPrune.violations).Count -ne 0) {
            $violations.Add("Cleanup prune evidence requires violations=0 but was $(@($cleanupDeferredPrune.violations).Count)")
        }

        foreach ($requiredCleanupPruneField in @('readyCount', 'blockedCount', 'prunedCount', 'remainingCount')) {
            if ($null -eq $cleanupDeferredPrune.$requiredCleanupPruneField) {
                $violations.Add("Cleanup prune evidence missing field: $requiredCleanupPruneField")
            }
        }

        if ($null -eq $cleanupDeferredPrune.apply) {
            $violations.Add('Cleanup prune evidence missing apply field')
        }
        elseif (-not [bool]$cleanupDeferredPrune.apply -and [int]$cleanupDeferredPrune.prunedCount -ne 0) {
            $violations.Add("Cleanup prune evidence apply=false requires prunedCount=0 but was $($cleanupDeferredPrune.prunedCount)")
        }

        $cleanupPruneGeneratedAt = [datetime]::MinValue
        if (-not [datetime]::TryParse("$($cleanupDeferredPrune.generatedAt)", [ref]$cleanupPruneGeneratedAt)) {
            $violations.Add('Cleanup prune evidence generatedAt parse failed')
        }
        else {
            $cleanupPruneAgeMinutes = ((Get-Date) - $cleanupPruneGeneratedAt).TotalMinutes
            if ($cleanupPruneAgeMinutes -gt $artifactFreshnessWindowMinutes) {
                $violations.Add("Cleanup prune evidence freshness breach: ageMinutes=$([math]::Round($cleanupPruneAgeMinutes, 2)) windowMinutes=$artifactFreshnessWindowMinutes")
            }
        }
    }
    catch {
        $violations.Add("Cleanup prune evidence parse failed: path=$cleanupDeferredPruneArtifactPath reason=$($_.Exception.Message)")
    }
}

$cleanupDeferredArtifactPath = Join-Path $evidenceDir 'cleanup_deferred_report.json'
if (Test-Path -LiteralPath $cleanupDeferredArtifactPath) {
    try {
        $cleanupDeferred = Get-Content -LiteralPath $cleanupDeferredArtifactPath -Raw -Encoding UTF8 | ConvertFrom-Json

        if ($null -eq $cleanupDeferred.violations) {
            $violations.Add('Cleanup deferred evidence missing violations field')
        }
        elseif (@($cleanupDeferred.violations).Count -ne 0) {
            $violations.Add("Cleanup deferred evidence requires violations=0 but was $(@($cleanupDeferred.violations).Count)")
        }

        if ($null -eq $cleanupDeferred.entryCount) {
            $violations.Add('Cleanup deferred evidence missing entryCount field')
        }
        elseif ([int]$cleanupDeferred.entryCount -ne 0) {
            $violations.Add("Cleanup deferred evidence requires entryCount=0 but was $($cleanupDeferred.entryCount)")
        }

        if ($null -eq $cleanupDeferred.pruneSummary) {
            $violations.Add('Cleanup deferred evidence missing pruneSummary field')
        }
        else {
            foreach ($requiredCleanupSummaryField in @('readyCount', 'blockedCount', 'prunedCount', 'remainingCount')) {
                if ($null -eq $cleanupDeferred.pruneSummary.$requiredCleanupSummaryField) {
                    $violations.Add("Cleanup deferred evidence pruneSummary missing field: $requiredCleanupSummaryField")
                }
            }
        }

        $cleanupDeferredGeneratedAt = [datetime]::MinValue
        if (-not [datetime]::TryParse("$($cleanupDeferred.generatedAt)", [ref]$cleanupDeferredGeneratedAt)) {
            $violations.Add('Cleanup deferred evidence generatedAt parse failed')
        }
        else {
            $cleanupDeferredAgeMinutes = ((Get-Date) - $cleanupDeferredGeneratedAt).TotalMinutes
            if ($cleanupDeferredAgeMinutes -gt $artifactFreshnessWindowMinutes) {
                $violations.Add("Cleanup deferred evidence freshness breach: ageMinutes=$([math]::Round($cleanupDeferredAgeMinutes, 2)) windowMinutes=$artifactFreshnessWindowMinutes")
            }
        }
    }
    catch {
        $violations.Add("Cleanup deferred evidence parse failed: path=$cleanupDeferredArtifactPath reason=$($_.Exception.Message)")
    }
}

$policyStatus = New-Object System.Collections.Generic.List[object]
foreach ($policy in $requiredPolicies) {
    $exists = Test-Path -LiteralPath $policy.Path
    $actualSchema = $null

    if ($exists) {
        try {
            $content = Get-Content -LiteralPath $policy.Path -Raw -Encoding UTF8 | ConvertFrom-Json
            $actualSchema = "$($content.schema)"
            if ($actualSchema -ne "$($policy.Schema)") {
                $violations.Add("Policy schema mismatch: path=$($policy.Path) expected=$($policy.Schema) actual=$actualSchema")
            }
        }
        catch {
            $violations.Add("Policy parse failed: path=$($policy.Path) reason=$($_.Exception.Message)")
        }
    }
    else {
        $violations.Add("Missing required plan policy file: $($policy.Path)")
    }

    $policyStatus.Add([ordered]@{
            path           = $policy.Path
            expectedSchema = $policy.Schema
            exists         = $exists
            actualSchema   = $actualSchema
        }) | Out-Null
}

$allowlistStatus = New-Object System.Collections.Generic.List[object]
foreach ($allowlist in $requiredAllowlists) {
    $exists = Test-Path -LiteralPath $allowlist.Path
    $actualSchema = $null
    $allowlistExpiry = $null

    if ($exists) {
        try {
            $content = Get-Content -LiteralPath $allowlist.Path -Raw -Encoding UTF8 | ConvertFrom-Json
            $actualSchema = "$($content.schema)"
            if ($actualSchema -ne "$($allowlist.Schema)") {
                $violations.Add("Allowlist schema mismatch: path=$($allowlist.Path) expected=$($allowlist.Schema) actual=$actualSchema")
            }

            foreach ($requiredField in @('owner', 'expiry', 'issue', 'rationale')) {
                if ($null -eq $content.$requiredField -or [string]::IsNullOrWhiteSpace("$($content.$requiredField)")) {
                    $violations.Add("Allowlist missing required field: path=$($allowlist.Path) field=$requiredField")
                }
            }

            $allowlistExpiryDate = [datetime]::MinValue
            if (-not [datetime]::TryParse("$($content.expiry)", [ref]$allowlistExpiryDate)) {
                $violations.Add("Allowlist expiry parse failed: path=$($allowlist.Path) value=$($content.expiry)")
            }
            else {
                $allowlistExpiry = $allowlistExpiryDate.ToString('o')
                if ((Get-Date) -gt $allowlistExpiryDate) {
                    $violations.Add("Allowlist expired: path=$($allowlist.Path) expiry=$($content.expiry)")
                }
            }
        }
        catch {
            $violations.Add("Allowlist parse failed: path=$($allowlist.Path) reason=$($_.Exception.Message)")
        }
    }
    else {
        $violations.Add("Missing required allowlist file: $($allowlist.Path)")
    }

    $allowlistStatus.Add([ordered]@{
            path           = $allowlist.Path
            expectedSchema = $allowlist.Schema
            exists         = $exists
            actualSchema   = $actualSchema
            expiry         = $allowlistExpiry
        }) | Out-Null
}

$backlogResidualArtifactPath = Join-Path $evidenceDir 'backlog_specfixed_residual_report.json'
$rollbackCompatibilityArtifactPath = Join-Path $evidenceDir 'rollback_compatibility_report.json'
if (Test-Path -LiteralPath $backlogResidualArtifactPath) {
    try {
        $backlogResidual = Get-Content -LiteralPath $backlogResidualArtifactPath -Raw -Encoding UTF8 | ConvertFrom-Json
        $actualBacklogPath = "$($backlogResidual.backlogPath)"

        if ([string]::IsNullOrWhiteSpace($actualBacklogPath) -or [System.IO.Path]::GetFullPath($actualBacklogPath) -ne $expectedBacklogPath) {
            $violations.Add("Backlog residual evidence backlogPath mismatch: expected=$expectedBacklogPath actual=$($backlogResidual.backlogPath)")
        }
        else {
            $backlogPathSatisfied = $true
        }

        if ($null -eq $backlogResidual.enforceNoSpecFixed -or -not [bool]$backlogResidual.enforceNoSpecFixed) {
            $violations.Add('Backlog residual evidence requires enforceNoSpecFixed=true')
        }

        if ($null -eq $backlogResidual.specFixedResidualCount -or [int]$backlogResidual.specFixedResidualCount -ne 0) {
            $violations.Add("Backlog residual evidence requires specFixedResidualCount=0 but was $($backlogResidual.specFixedResidualCount)")
        }
    }
    catch {
        $violations.Add("Backlog residual evidence parse failed: path=$backlogResidualArtifactPath reason=$($_.Exception.Message)")
    }
}

$completionDefinitionStatus = [ordered]@{
    bridgeRuntimeControllable = [ordered]@{
        clause = 'bridge runtime が制御不能化しない'
        satisfied = $false
        reason = ''
        evidenceLocators = @(
            "${triggerAuditArtifactPath}:policyViolations:completion_bridge_controllable_policy",
            "${triggerAuditArtifactPath}:metrics.runtimeExecutionViewUsageCount:completion_bridge_controllable_runtime_view",
            "${ownershipMigrationArtifactPath}:allStepsSatisfied:completion_bridge_controllable_ownership"
        )
    }
    ciDeviationDetectable = [ordered]@{
        clause = 'CIで逸脱を機械検知できる'
        satisfied = $false
        reason = ''
        evidenceLocators = @(
            "${tierRunnerPath}:scriptsToRun:completion_ci_detectable_tier_runner",
            "${triggerAuditArtifactPath}:schema:completion_ci_detectable_trigger_schema",
            "${reportPath}:schema:completion_ci_detectable_completeness_schema"
        )
    }
    rollbackSubsystemGranularity = [ordered]@{
        clause = 'rollback が subsystem 粒度で機能する'
        satisfied = $false
        reason = ''
        evidenceLocators = @(
            "${rollbackCompatibilityArtifactPath}:violations:completion_rollback_violations",
            "${rollbackCompatibilityArtifactPath}:scenarios.missingFlags:completion_rollback_missing_flags",
            "${rollbackCompatibilityArtifactPath}:scenarios.expired:completion_rollback_expired"
        )
    }
    deferredTriggerConvergence = [ordered]@{
        clause = 'deferred が trigger 駆動で確実に縮退する'
        satisfied = $false
        reason = ''
        evidenceLocators = @(
            "${triggerCleanupReadinessArtifactPath}:readyCount:completion_deferred_ready",
            "${triggerCleanupCompletionArtifactPath}:deferredRegistryEntryCount:completion_deferred_registry_count",
            "${backlogResidualArtifactPath}:specFixedResidualCount:completion_deferred_specfixed"
        )
    }
    noMajorIncidentIncrease = [ordered]@{
        clause = '主要事故（UAF/RT blocking/partial visibility/stale access）が実運用で増加しない'
        satisfied = $false
        reason = ''
        evidenceLocators = @(
            "${canaryNormalizationArtifactPath}:violations:completion_incident_canary_violations",
            "${metricGovernanceArtifactPath}:normalizationPolicy.strictModeRequireAllMetrics:completion_incident_metric_strict",
            "${triggerAuditArtifactPath}:policyViolations:completion_incident_policy_violations"
        )
    }
}

$requiredCompletionLocatorLabels = @{
    bridgeRuntimeControllable = @('completion_bridge_controllable_policy', 'completion_bridge_controllable_runtime_view', 'completion_bridge_controllable_ownership')
    ciDeviationDetectable = @('completion_ci_detectable_tier_runner', 'completion_ci_detectable_trigger_schema', 'completion_ci_detectable_completeness_schema')
    rollbackSubsystemGranularity = @('completion_rollback_violations', 'completion_rollback_missing_flags', 'completion_rollback_expired')
    deferredTriggerConvergence = @('completion_deferred_ready', 'completion_deferred_registry_count', 'completion_deferred_specfixed')
    noMajorIncidentIncrease = @('completion_incident_canary_violations', 'completion_incident_metric_strict', 'completion_incident_policy_violations')
}

$allowedCompletionLocatorPrefixes = @(
    "${triggerAuditArtifactPath}:",
    "${tierRunnerPath}:",
    "${reportPath}:",
    "${rollbackCompatibilityArtifactPath}:",
    "${triggerCleanupReadinessArtifactPath}:",
    "${triggerCleanupCompletionArtifactPath}:",
    "${backlogResidualArtifactPath}:",
    "${canaryNormalizationArtifactPath}:",
    "${metricGovernanceArtifactPath}:",
    "${ownershipMigrationArtifactPath}:"
)

foreach ($completionKey in @('bridgeRuntimeControllable', 'ciDeviationDetectable', 'rollbackSubsystemGranularity', 'deferredTriggerConvergence', 'noMajorIncidentIncrease')) {
    # completion definition matrix validation is executed after phase evidence is computed.
}

$phaseCompletionMatrix = [ordered]@{
    phase0 = [ordered]@{
        title = '統制基盤の先行導入'
        satisfied = $false
        checks = @('trigger_audit.activeDspRefCount=0', 'trigger_audit.runtimeExecutionViewUsageCount=0', 'trigger_audit.policyViolations=0')
        reason = ''
        evidenceLocators = @()
    }
    phase1 = [ordered]@{
        title = '計測可能トリガー化'
        satisfied = $false
        checks = @('trigger_audit.policyEvaluations exists', 'trigger_cleanup_readiness.readyCount=0', 'trigger_cleanup_readiness.blockedCount=0')
        reason = ''
        evidenceLocators = @()
    }
    phase2 = [ordered]@{
        title = 'enforcement 高度化（grep→AST）'
        satisfied = $false
        checks = @('enforcement_adoption.withinTarget=true', 'advancedCoverageRatio>=minAdvancedCoverageRatio', 'enforcement_source_purity violations=0')
        reason = ''
        evidenceLocators = @()
    }
    phase3 = [ordered]@{
        title = 'facade 統制'
        satisfied = $false
        checks = @('trigger_audit.retireFacadeDirectDependencyCount=0', 'trigger_audit.retireFacadeRuntimeExecutionCount=0', 'facade_bypass.violations=0')
        reason = ''
        evidenceLocators = @()
    }
    phase4 = [ordered]@{
        title = 'crossfade 専用移行'
        satisfied = $false
        checks = @('canary_baseline_normalization.violations=0', 'phase4_generation_drift.violations=0')
        reason = ''
        evidenceLocators = @()
    }
    phase5 = [ordered]@{
        title = 'rollback hierarchy 導入'
        satisfied = $false
        checks = @('rollback_compatibility.violations=0', 'rollback scenarios missingFlags=0', 'rollback scenarios expired=false')
        reason = ''
        evidenceLocators = @()
    }
    phase6 = [ordered]@{
        title = 'cleanup（trigger達成後）'
        satisfied = $false
        checks = @('trigger_cleanup_completion.cleanupCompleted=true', 'trigger_cleanup_completion.deferredRegistryEntryCount=0', 'backlog_specfixed_residual.specFixedResidualCount=0')
        reason = ''
        evidenceLocators = @()
    }
}

$triggerAuditForPhaseMatrix = $null
if (Test-Path -LiteralPath $triggerAuditArtifactPath) {
    try {
        $triggerAuditForPhaseMatrix = Get-Content -LiteralPath $triggerAuditArtifactPath -Raw -Encoding UTF8 | ConvertFrom-Json
    }
    catch {
        $violations.Add("Phase completion trigger audit parse failed: path=$triggerAuditArtifactPath reason=$($_.Exception.Message)")
    }
}

$triggerReadinessForPhaseMatrix = $null
if (Test-Path -LiteralPath $triggerCleanupReadinessArtifactPath) {
    try {
        $triggerReadinessForPhaseMatrix = Get-Content -LiteralPath $triggerCleanupReadinessArtifactPath -Raw -Encoding UTF8 | ConvertFrom-Json
    }
    catch {
        $violations.Add("Phase completion readiness parse failed: path=$triggerCleanupReadinessArtifactPath reason=$($_.Exception.Message)")
    }
}

$enforcementAdoptionForPhaseMatrix = $null
if (Test-Path -LiteralPath $enforcementAdoptionArtifactPath) {
    try {
        $enforcementAdoptionForPhaseMatrix = Get-Content -LiteralPath $enforcementAdoptionArtifactPath -Raw -Encoding UTF8 | ConvertFrom-Json
    }
    catch {
        $violations.Add("Phase completion enforcement adoption parse failed: path=$enforcementAdoptionArtifactPath reason=$($_.Exception.Message)")
    }
}

$enforcementPurityForPhaseMatrix = $null
if (Test-Path -LiteralPath $enforcementSourcePurityArtifactPath) {
    try {
        $enforcementPurityForPhaseMatrix = Get-Content -LiteralPath $enforcementSourcePurityArtifactPath -Raw -Encoding UTF8 | ConvertFrom-Json
    }
    catch {
        $violations.Add("Phase completion enforcement purity parse failed: path=$enforcementSourcePurityArtifactPath reason=$($_.Exception.Message)")
    }
}

$facadeBypassArtifactPath = Join-Path $evidenceDir 'facade_bypass_report.json'
$facadeBypassForPhaseMatrix = $null
if (Test-Path -LiteralPath $facadeBypassArtifactPath) {
    try {
        $facadeBypassForPhaseMatrix = Get-Content -LiteralPath $facadeBypassArtifactPath -Raw -Encoding UTF8 | ConvertFrom-Json
    }
    catch {
        $violations.Add("Phase completion facade bypass parse failed: path=$facadeBypassArtifactPath reason=$($_.Exception.Message)")
    }
}

$canaryForPhaseMatrix = $null
if (Test-Path -LiteralPath $canaryNormalizationArtifactPath) {
    try {
        $canaryForPhaseMatrix = Get-Content -LiteralPath $canaryNormalizationArtifactPath -Raw -Encoding UTF8 | ConvertFrom-Json
    }
    catch {
        $violations.Add("Phase completion canary parse failed: path=$canaryNormalizationArtifactPath reason=$($_.Exception.Message)")
    }
}

$phase4DriftArtifactPath = Join-Path $evidenceDir 'phase4_generation_drift_report.json'
$phase4DriftForPhaseMatrix = $null
if (Test-Path -LiteralPath $phase4DriftArtifactPath) {
    try {
        $phase4DriftForPhaseMatrix = Get-Content -LiteralPath $phase4DriftArtifactPath -Raw -Encoding UTF8 | ConvertFrom-Json
    }
    catch {
        $violations.Add("Phase completion phase4 drift parse failed: path=$phase4DriftArtifactPath reason=$($_.Exception.Message)")
    }
}

$rollbackForPhaseMatrix = $null
$rollbackCompatibilityArtifactPath = Join-Path $evidenceDir 'rollback_compatibility_report.json'
if (Test-Path -LiteralPath $rollbackCompatibilityArtifactPath) {
    try {
        $rollbackForPhaseMatrix = Get-Content -LiteralPath $rollbackCompatibilityArtifactPath -Raw -Encoding UTF8 | ConvertFrom-Json
    }
    catch {
        $violations.Add("Phase completion rollback parse failed: path=$rollbackCompatibilityArtifactPath reason=$($_.Exception.Message)")
    }
}

$cleanupCompletionForPhaseMatrix = $null
if (Test-Path -LiteralPath $triggerCleanupCompletionArtifactPath) {
    try {
        $cleanupCompletionForPhaseMatrix = Get-Content -LiteralPath $triggerCleanupCompletionArtifactPath -Raw -Encoding UTF8 | ConvertFrom-Json
    }
    catch {
        $violations.Add("Phase completion cleanup completion parse failed: path=$triggerCleanupCompletionArtifactPath reason=$($_.Exception.Message)")
    }
}

$backlogResidualForPhaseMatrix = $null
if (Test-Path -LiteralPath $backlogResidualArtifactPath) {
    try {
        $backlogResidualForPhaseMatrix = Get-Content -LiteralPath $backlogResidualArtifactPath -Raw -Encoding UTF8 | ConvertFrom-Json
    }
    catch {
        $violations.Add("Phase completion backlog residual parse failed: path=$backlogResidualArtifactPath reason=$($_.Exception.Message)")
    }
}

$phase0Satisfied = $false
if ($null -ne $triggerAuditForPhaseMatrix -and $null -ne $triggerAuditForPhaseMatrix.metrics -and $null -ne $triggerAuditForPhaseMatrix.policyViolations) {
    $phase0Satisfied =
        ([int]$triggerAuditForPhaseMatrix.metrics.activeDspRefCount -eq 0) -and
        ([int]$triggerAuditForPhaseMatrix.metrics.runtimeExecutionViewUsageCount -eq 0) -and
        (@($triggerAuditForPhaseMatrix.policyViolations).Count -eq 0)
}
$phaseCompletionMatrix.phase0.satisfied = $phase0Satisfied
$phaseCompletionMatrix.phase0.reason = if ($phase0Satisfied) { 'Phase 0 controls are satisfied by trigger audit zero-metrics and zero policy violations.' } else { 'Phase 0 controls are not fully satisfied by trigger audit evidence.' }
$phaseCompletionMatrix.phase0.evidenceLocators = @(
    "${triggerAuditArtifactPath}:metrics.activeDspRefCount:phase0_active_dsp",
    "${triggerAuditArtifactPath}:metrics.runtimeExecutionViewUsageCount:phase0_runtime_view",
    "${triggerAuditArtifactPath}:policyViolations:phase0_policy_violations"
)
if (-not $phase0Satisfied) {
    $violations.Add('Phase completion contract violated: Phase 0 (統制基盤の先行導入) is not satisfied')
}

$phase1Satisfied = $false
if ($null -ne $triggerAuditForPhaseMatrix -and $null -ne $triggerAuditForPhaseMatrix.policyEvaluations -and $null -ne $triggerReadinessForPhaseMatrix) {
    $phase1Satisfied =
        (@($triggerAuditForPhaseMatrix.policyEvaluations).Count -gt 0) -and
        ([int]$triggerReadinessForPhaseMatrix.readyCount -eq 0) -and
        ([int]$triggerReadinessForPhaseMatrix.blockedCount -eq 0)
}
$phaseCompletionMatrix.phase1.satisfied = $phase1Satisfied
$phaseCompletionMatrix.phase1.reason = if ($phase1Satisfied) { 'Phase 1 trigger readiness is confirmed by policy evaluations and cleanup readiness counts.' } else { 'Phase 1 trigger readiness is incomplete (policy evaluations/readiness counters mismatch).' }
$phaseCompletionMatrix.phase1.evidenceLocators = @(
    "${triggerAuditArtifactPath}:policyEvaluations:phase1_policy_evaluations",
    "${triggerCleanupReadinessArtifactPath}:readyCount:phase1_ready_count",
    "${triggerCleanupReadinessArtifactPath}:blockedCount:phase1_blocked_count"
)
if (-not $phase1Satisfied) {
    $violations.Add('Phase completion contract violated: Phase 1 (計測可能トリガー化) is not satisfied')
}

$phase2Satisfied = $false
if ($null -ne $enforcementAdoptionForPhaseMatrix -and $null -ne $enforcementPurityForPhaseMatrix) {
    $coverageRatio = [double]$enforcementAdoptionForPhaseMatrix.advancedCoverageRatio
    $coverageMin = [double]$enforcementAdoptionForPhaseMatrix.minAdvancedCoverageRatio
    $phase2Satisfied =
        ([bool]$enforcementAdoptionForPhaseMatrix.withinTarget) -and
        ($coverageRatio -ge $coverageMin) -and
        ([int]$enforcementPurityForPhaseMatrix.rawRegexViolationCount -eq 0) -and
        ([int]$enforcementPurityForPhaseMatrix.unknownViolationCount -eq 0)
}
$phaseCompletionMatrix.phase2.satisfied = $phase2Satisfied
$phaseCompletionMatrix.phase2.reason = if ($phase2Satisfied) { 'Phase 2 enforcement upgrade is confirmed by adoption target and purity zero-violation evidence.' } else { 'Phase 2 enforcement upgrade is incomplete (adoption ratio/withinTarget/purity mismatch).' }
$phaseCompletionMatrix.phase2.evidenceLocators = @(
    "${enforcementAdoptionArtifactPath}:withinTarget:phase2_within_target",
    "${enforcementAdoptionArtifactPath}:advancedCoverageRatio:phase2_coverage_ratio",
    "${enforcementSourcePurityArtifactPath}:rawRegexViolationCount:phase2_raw_regex",
    "${enforcementSourcePurityArtifactPath}:unknownViolationCount:phase2_unknown_violations"
)
if (-not $phase2Satisfied) {
    $violations.Add('Phase completion contract violated: Phase 2 (enforcement 高度化) is not satisfied')
}

$phase3Satisfied = $false
if ($null -ne $triggerAuditForPhaseMatrix -and $null -ne $triggerAuditForPhaseMatrix.metrics -and $null -ne $facadeBypassForPhaseMatrix) {
    $phase3Satisfied =
        ([int]$triggerAuditForPhaseMatrix.metrics.retireFacadeDirectDependencyCount -eq 0) -and
        ([int]$triggerAuditForPhaseMatrix.metrics.retireFacadeRuntimeExecutionCount -eq 0) -and
        (@($facadeBypassForPhaseMatrix.violations).Count -eq 0)
}
$phaseCompletionMatrix.phase3.satisfied = $phase3Satisfied
$phaseCompletionMatrix.phase3.reason = if ($phase3Satisfied) { 'Phase 3 facade control is satisfied by zero direct/runtime dependency metrics and facade bypass violations.' } else { 'Phase 3 facade control is incomplete (retire facade metric/bypass violations mismatch).' }
$phaseCompletionMatrix.phase3.evidenceLocators = @(
    "${triggerAuditArtifactPath}:metrics.retireFacadeDirectDependencyCount:phase3_direct_dependency",
    "${triggerAuditArtifactPath}:metrics.retireFacadeRuntimeExecutionCount:phase3_runtime_execution",
    "${facadeBypassArtifactPath}:violations:phase3_facade_violations"
)
if (-not $phase3Satisfied) {
    $violations.Add('Phase completion contract violated: Phase 3 (facade 統制) is not satisfied')
}

$phase4Satisfied = $false
if ($null -ne $canaryForPhaseMatrix -and $null -ne $phase4DriftForPhaseMatrix) {
    $phase4Satisfied =
        (@($canaryForPhaseMatrix.violations).Count -eq 0) -and
        (@($phase4DriftForPhaseMatrix.violations).Count -eq 0)
}
$phaseCompletionMatrix.phase4.satisfied = $phase4Satisfied
$phaseCompletionMatrix.phase4.reason = if ($phase4Satisfied) { 'Phase 4 crossfade migration is satisfied by canary normalization and generation drift zero-violation evidence.' } else { 'Phase 4 crossfade migration is incomplete (canary/drift violations detected).' }
$phaseCompletionMatrix.phase4.evidenceLocators = @(
    "${canaryNormalizationArtifactPath}:violations:phase4_canary_violations",
    "${phase4DriftArtifactPath}:violations:phase4_generation_drift"
)
if (-not $phase4Satisfied) {
    $violations.Add('Phase completion contract violated: Phase 4 (crossfade 専用移行) is not satisfied')
}

$phase5Satisfied = $false
if ($null -ne $rollbackForPhaseMatrix) {
    $phase5Satisfied =
        (@($rollbackForPhaseMatrix.violations).Count -eq 0) -and
        (@($rollbackForPhaseMatrix.scenarios | Where-Object { @($_.missingFlags).Count -gt 0 }).Count -eq 0) -and
        (@($rollbackForPhaseMatrix.scenarios | Where-Object { [bool]$_.expired }).Count -eq 0)
}
$phaseCompletionMatrix.phase5.satisfied = $phase5Satisfied
$phaseCompletionMatrix.phase5.reason = if ($phase5Satisfied) { 'Phase 5 rollback hierarchy is satisfied by rollback compatibility evidence with no missing flags/expired scenarios.' } else { 'Phase 5 rollback hierarchy is incomplete (rollback violations/missing flags/expired scenarios detected).' }
$phaseCompletionMatrix.phase5.evidenceLocators = @(
    "${rollbackCompatibilityArtifactPath}:violations:phase5_rollback_violations",
    "${rollbackCompatibilityArtifactPath}:scenarios.missingFlags:phase5_missing_flags",
    "${rollbackCompatibilityArtifactPath}:scenarios.expired:phase5_expired"
)
if (-not $phase5Satisfied) {
    $violations.Add('Phase completion contract violated: Phase 5 (rollback hierarchy 導入) is not satisfied')
}

$phase6Satisfied = $false
if ($null -ne $cleanupCompletionForPhaseMatrix -and $null -ne $backlogResidualForPhaseMatrix) {
    $phase6Satisfied =
        ([bool]$cleanupCompletionForPhaseMatrix.cleanupCompleted) -and
        ([int]$cleanupCompletionForPhaseMatrix.deferredRegistryEntryCount -eq 0) -and
        ([int]$backlogResidualForPhaseMatrix.specFixedResidualCount -eq 0)
}
$phaseCompletionMatrix.phase6.satisfied = $phase6Satisfied
$phaseCompletionMatrix.phase6.reason = if ($phase6Satisfied) { 'Phase 6 cleanup is satisfied by cleanup completion and backlog residual zero evidence.' } else { 'Phase 6 cleanup is incomplete (cleanup completion/deferred entries/backlog residual mismatch).' }
$phaseCompletionMatrix.phase6.evidenceLocators = @(
    "${triggerCleanupCompletionArtifactPath}:cleanupCompleted:phase6_cleanup_completed",
    "${triggerCleanupCompletionArtifactPath}:deferredRegistryEntryCount:phase6_deferred_count",
    "${backlogResidualArtifactPath}:specFixedResidualCount:phase6_backlog_residual"
)
if (-not $phase6Satisfied) {
    $violations.Add('Phase completion contract violated: Phase 6 (cleanup) is not satisfied')
}

$requiredPhaseLocatorLabels = @{
    phase0 = @('phase0_active_dsp', 'phase0_runtime_view', 'phase0_policy_violations')
    phase1 = @('phase1_policy_evaluations', 'phase1_ready_count', 'phase1_blocked_count')
    phase2 = @('phase2_within_target', 'phase2_coverage_ratio', 'phase2_raw_regex', 'phase2_unknown_violations')
    phase3 = @('phase3_direct_dependency', 'phase3_runtime_execution', 'phase3_facade_violations')
    phase4 = @('phase4_canary_violations', 'phase4_generation_drift')
    phase5 = @('phase5_rollback_violations', 'phase5_missing_flags', 'phase5_expired')
    phase6 = @('phase6_cleanup_completed', 'phase6_deferred_count', 'phase6_backlog_residual')
}

$requiredPhaseChecks = @{
    phase0 = @('trigger_audit.activeDspRefCount=0', 'trigger_audit.runtimeExecutionViewUsageCount=0', 'trigger_audit.policyViolations=0')
    phase1 = @('trigger_audit.policyEvaluations exists', 'trigger_cleanup_readiness.readyCount=0', 'trigger_cleanup_readiness.blockedCount=0')
    phase2 = @('enforcement_adoption.withinTarget=true', 'advancedCoverageRatio>=minAdvancedCoverageRatio', 'enforcement_source_purity violations=0')
    phase3 = @('trigger_audit.retireFacadeDirectDependencyCount=0', 'trigger_audit.retireFacadeRuntimeExecutionCount=0', 'facade_bypass.violations=0')
    phase4 = @('canary_baseline_normalization.violations=0', 'phase4_generation_drift.violations=0')
    phase5 = @('rollback_compatibility.violations=0', 'rollback scenarios missingFlags=0', 'rollback scenarios expired=false')
    phase6 = @('trigger_cleanup_completion.cleanupCompleted=true', 'trigger_cleanup_completion.deferredRegistryEntryCount=0', 'backlog_specfixed_residual.specFixedResidualCount=0')
}

$phaseExpectedSatisfied = @{
    phase0 = $phase0Satisfied
    phase1 = $phase1Satisfied
    phase2 = $phase2Satisfied
    phase3 = $phase3Satisfied
    phase4 = $phase4Satisfied
    phase5 = $phase5Satisfied
    phase6 = $phase6Satisfied
}

$phaseContractViolationByKey = @{
    phase0 = 'Phase completion contract violated: Phase 0 (統制基盤の先行導入) is not satisfied'
    phase1 = 'Phase completion contract violated: Phase 1 (計測可能トリガー化) is not satisfied'
    phase2 = 'Phase completion contract violated: Phase 2 (enforcement 高度化) is not satisfied'
    phase3 = 'Phase completion contract violated: Phase 3 (facade 統制) is not satisfied'
    phase4 = 'Phase completion contract violated: Phase 4 (crossfade 専用移行) is not satisfied'
    phase5 = 'Phase completion contract violated: Phase 5 (rollback hierarchy 導入) is not satisfied'
    phase6 = 'Phase completion contract violated: Phase 6 (cleanup) is not satisfied'
}

$allowedPhaseLocatorPrefixes = @(
    "${triggerAuditArtifactPath}:",
    "${triggerCleanupReadinessArtifactPath}:",
    "${enforcementAdoptionArtifactPath}:",
    "${enforcementSourcePurityArtifactPath}:",
    "${facadeBypassArtifactPath}:",
    "${canaryNormalizationArtifactPath}:",
    "${phase4DriftArtifactPath}:",
    "${rollbackCompatibilityArtifactPath}:",
    "${triggerCleanupCompletionArtifactPath}:",
    "${backlogResidualArtifactPath}:"
)

foreach ($phaseKey in @('phase0', 'phase1', 'phase2', 'phase3', 'phase4', 'phase5', 'phase6')) {
    $phaseEntry = $phaseCompletionMatrix[$phaseKey]

    if ($null -eq $phaseEntry.title -or [string]::IsNullOrWhiteSpace("$($phaseEntry.title)")) {
        $violations.Add("Phase completion matrix contract violated: $phaseKey missing non-empty title")
    }

    if ($null -eq $phaseEntry.checks -or @($phaseEntry.checks).Count -eq 0) {
        $violations.Add("Phase completion matrix contract violated: $phaseKey missing checks")
    }
    elseif ($requiredPhaseChecks.ContainsKey($phaseKey)) {
        $phaseChecks = @($phaseEntry.checks)
        $phaseRequiredChecks = @($requiredPhaseChecks[$phaseKey])

        $phaseCheckSet = New-Object 'System.Collections.Generic.HashSet[string]' ([System.StringComparer]::Ordinal)
        $phaseCheckHasDuplicate = $false
        foreach ($phaseCheck in $phaseChecks) {
            $phaseCheckText = "$phaseCheck"
            if (-not $phaseCheckSet.Add($phaseCheckText)) {
                $phaseCheckHasDuplicate = $true
            }
        }

        if ($phaseCheckHasDuplicate) {
            $violations.Add("Phase completion matrix contract violated: $phaseKey contains duplicate checks")
        }

        foreach ($requiredPhaseCheck in $phaseRequiredChecks) {
            if ($phaseChecks -notcontains $requiredPhaseCheck) {
                $violations.Add("Phase completion matrix contract violated: $phaseKey missing required check: $requiredPhaseCheck")
            }
        }

        foreach ($phaseCheck in $phaseChecks) {
            if ($phaseRequiredChecks -notcontains "$phaseCheck") {
                $violations.Add("Phase completion matrix contract violated: $phaseKey has unexpected check: $phaseCheck")
            }
        }
    }

    if ($null -eq $phaseEntry.satisfied) {
        $violations.Add("Phase completion matrix contract violated: $phaseKey missing satisfied field")
    }
    elseif ($phaseExpectedSatisfied.ContainsKey($phaseKey) -and ([bool]$phaseEntry.satisfied -ne [bool]$phaseExpectedSatisfied[$phaseKey])) {
        $violations.Add("Phase completion matrix contract violated: $phaseKey satisfied mismatch with computed phase state")
    }

    if ($phaseContractViolationByKey.ContainsKey($phaseKey)) {
        $requiredPhaseContractViolation = $phaseContractViolationByKey[$phaseKey]
        if (-not [bool]$phaseEntry.satisfied -and -not ($violations -contains $requiredPhaseContractViolation)) {
            $violations.Add("Phase completion matrix contract violated: $phaseKey missing unsatisfied phase contract violation")
        }
    }

    if ($null -eq $phaseEntry.reason -or [string]::IsNullOrWhiteSpace("$($phaseEntry.reason)")) {
        $violations.Add("Phase completion matrix contract violated: $phaseKey missing non-empty reason")
    }

    if ($null -eq $phaseEntry.evidenceLocators -or @($phaseEntry.evidenceLocators).Count -eq 0) {
        $violations.Add("Phase completion matrix contract violated: $phaseKey missing evidenceLocators")
        continue
    }

    foreach ($phaseLocator in @($phaseEntry.evidenceLocators)) {
        if ([string]::IsNullOrWhiteSpace("$phaseLocator")) {
            $violations.Add("Phase completion matrix contract violated: $phaseKey contains empty evidence locator")
            continue
        }

        if ("$phaseLocator" -notmatch '^.+:.+:.+$') {
            $violations.Add("Phase completion matrix contract violated: $phaseKey evidence locator format invalid: locator=$phaseLocator")
        }

        $phasePrefixSatisfied = $false
        foreach ($allowedPhaseLocatorPrefix in $allowedPhaseLocatorPrefixes) {
            if ("$phaseLocator".StartsWith($allowedPhaseLocatorPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
                $phasePrefixSatisfied = $true
                break
            }
        }

        if (-not $phasePrefixSatisfied) {
            $violations.Add("Phase completion matrix contract violated: $phaseKey evidence locator prefix invalid: locator=$phaseLocator")
        }
    }

    if ($requiredPhaseLocatorLabels.ContainsKey($phaseKey)) {
        foreach ($requiredPhaseLocatorLabel in @($requiredPhaseLocatorLabels[$phaseKey])) {
            $phaseLabelFound = $false
            foreach ($phaseLocator in @($phaseEntry.evidenceLocators)) {
                if ("$phaseLocator" -like "*:$requiredPhaseLocatorLabel") {
                    $phaseLabelFound = $true
                    break
                }
            }

            if (-not $phaseLabelFound) {
                $violations.Add("Phase completion matrix contract violated: $phaseKey missing required locator label: $requiredPhaseLocatorLabel")
            }
        }
    }
}

$ownershipForCompletion = $null
if (Test-Path -LiteralPath $ownershipMigrationArtifactPath) {
    try {
        $ownershipForCompletion = Get-Content -LiteralPath $ownershipMigrationArtifactPath -Raw -Encoding UTF8 | ConvertFrom-Json
    }
    catch {
        $violations.Add("Completion definition ownership evidence parse failed: path=$ownershipMigrationArtifactPath reason=$($_.Exception.Message)")
    }
}

$metricGovernanceForCompletion = $null
if (Test-Path -LiteralPath $metricGovernanceArtifactPath) {
    try {
        $metricGovernanceForCompletion = Get-Content -LiteralPath $metricGovernanceArtifactPath -Raw -Encoding UTF8 | ConvertFrom-Json
    }
    catch {
        $violations.Add("Completion definition metric governance parse failed: path=$metricGovernanceArtifactPath reason=$($_.Exception.Message)")
    }
}

$completionBridgeRuntimeControllableSatisfied =
    ($null -ne $triggerAuditForPhaseMatrix) -and
    ($null -ne $triggerAuditForPhaseMatrix.policyViolations) -and
    (@($triggerAuditForPhaseMatrix.policyViolations).Count -eq 0) -and
    ($null -ne $triggerAuditForPhaseMatrix.metrics) -and
    ([int]$triggerAuditForPhaseMatrix.metrics.runtimeExecutionViewUsageCount -eq 0) -and
    ($null -ne $ownershipForCompletion) -and
    ([bool]$ownershipForCompletion.allStepsSatisfied)
$completionDefinitionStatus.bridgeRuntimeControllable.satisfied = $completionBridgeRuntimeControllableSatisfied
$completionDefinitionStatus.bridgeRuntimeControllable.reason = if ($completionBridgeRuntimeControllableSatisfied) { 'Trigger policy violations are zero, runtime view usage is converged, and ownership transfer is complete.' } else { 'Trigger/ownership convergence evidence does not satisfy bridge runtime controllability clause.' }
if (-not $completionBridgeRuntimeControllableSatisfied) {
    $violations.Add('Completion definition contract violated: bridgeRuntimeControllable is not satisfied')
}

$completionCiDeviationDetectableSatisfied =
    ($requiredScripts.Count -gt 0) -and
    ($scriptOrderSatisfied) -and
    (Test-Path -LiteralPath $triggerAuditArtifactPath) -and
    (@($artifactStatus | Where-Object { [bool]$_.exists }).Count -ge 10)
$completionDefinitionStatus.ciDeviationDetectable.satisfied = $completionCiDeviationDetectableSatisfied
$completionDefinitionStatus.ciDeviationDetectable.reason = if ($completionCiDeviationDetectableSatisfied) { 'Tier runner wiring and generated evidence artifacts confirm CI-detectable deviation contracts.' } else { 'Tier runner/evidence generation contracts are incomplete for CI-detectable deviation clause.' }
if (-not $completionCiDeviationDetectableSatisfied) {
    $violations.Add('Completion definition contract violated: ciDeviationDetectable is not satisfied')
}

$completionRollbackSubsystemGranularitySatisfied = $phase5Satisfied
$completionDefinitionStatus.rollbackSubsystemGranularity.satisfied = $completionRollbackSubsystemGranularitySatisfied
$completionDefinitionStatus.rollbackSubsystemGranularity.reason = if ($completionRollbackSubsystemGranularitySatisfied) { 'Rollback compatibility matrix shows no violations, missing flags, or expired scenarios.' } else { 'Rollback compatibility evidence does not satisfy subsystem granularity clause.' }
if (-not $completionRollbackSubsystemGranularitySatisfied) {
    $violations.Add('Completion definition contract violated: rollbackSubsystemGranularity is not satisfied')
}

$completionDeferredTriggerConvergenceSatisfied = $phase6Satisfied -and $phase1Satisfied
$completionDefinitionStatus.deferredTriggerConvergence.satisfied = $completionDeferredTriggerConvergenceSatisfied
$completionDefinitionStatus.deferredTriggerConvergence.reason = if ($completionDeferredTriggerConvergenceSatisfied) { 'Trigger readiness/completion and spec-fixed backlog residual are all converged to zero.' } else { 'Deferred cleanup trigger convergence evidence is incomplete.' }
if (-not $completionDeferredTriggerConvergenceSatisfied) {
    $violations.Add('Completion definition contract violated: deferredTriggerConvergence is not satisfied')
}

$completionNoMajorIncidentIncreaseSatisfied =
    ($phase4Satisfied) -and
    ($null -ne $metricGovernanceForCompletion) -and
    ($null -ne $metricGovernanceForCompletion.normalizationPolicy) -and
    ([bool]$metricGovernanceForCompletion.normalizationPolicy.strictModeRequireAllMetrics) -and
    ($null -ne $triggerAuditForPhaseMatrix) -and
    (@($triggerAuditForPhaseMatrix.policyViolations).Count -eq 0)
$completionDefinitionStatus.noMajorIncidentIncrease.satisfied = $completionNoMajorIncidentIncreaseSatisfied
$completionDefinitionStatus.noMajorIncidentIncrease.reason = if ($completionNoMajorIncidentIncreaseSatisfied) { 'Canary normalization/metric strict-mode and trigger policy evidence show no major incident increase indicators.' } else { 'Major incident non-increase clause is not fully supported by canary/metric/trigger evidence.' }
if (-not $completionNoMajorIncidentIncreaseSatisfied) {
    $violations.Add('Completion definition contract violated: noMajorIncidentIncrease is not satisfied')
}

foreach ($completionKey in @('bridgeRuntimeControllable', 'ciDeviationDetectable', 'rollbackSubsystemGranularity', 'deferredTriggerConvergence', 'noMajorIncidentIncrease')) {
    $completionEntry = $completionDefinitionStatus[$completionKey]

    if ($null -eq $completionEntry.clause -or [string]::IsNullOrWhiteSpace("$($completionEntry.clause)")) {
        $violations.Add("Completion definition matrix contract violated: $completionKey missing clause")
    }

    if ($null -eq $completionEntry.satisfied) {
        $violations.Add("Completion definition matrix contract violated: $completionKey missing satisfied field")
    }

    if ($null -eq $completionEntry.reason -or [string]::IsNullOrWhiteSpace("$($completionEntry.reason)")) {
        $violations.Add("Completion definition matrix contract violated: $completionKey missing non-empty reason")
    }

    if ($null -eq $completionEntry.evidenceLocators -or @($completionEntry.evidenceLocators).Count -eq 0) {
        $violations.Add("Completion definition matrix contract violated: $completionKey missing evidenceLocators")
        continue
    }

    foreach ($completionLocator in @($completionEntry.evidenceLocators)) {
        if ([string]::IsNullOrWhiteSpace("$completionLocator")) {
            $violations.Add("Completion definition matrix contract violated: $completionKey contains empty evidence locator")
            continue
        }

        if ("$completionLocator" -notmatch '^.+:.+:.+$') {
            $violations.Add("Completion definition matrix contract violated: $completionKey evidence locator format invalid: locator=$completionLocator")
        }

        $completionPrefixSatisfied = $false
        foreach ($allowedCompletionLocatorPrefix in $allowedCompletionLocatorPrefixes) {
            if ("$completionLocator".StartsWith($allowedCompletionLocatorPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
                $completionPrefixSatisfied = $true
                break
            }
        }

        if (-not $completionPrefixSatisfied) {
            $violations.Add("Completion definition matrix contract violated: $completionKey evidence locator prefix invalid: locator=$completionLocator")
        }
    }

    if ($requiredCompletionLocatorLabels.ContainsKey($completionKey)) {
        foreach ($requiredCompletionLocatorLabel in @($requiredCompletionLocatorLabels[$completionKey])) {
            $completionLabelFound = $false
            foreach ($completionLocator in @($completionEntry.evidenceLocators)) {
                if ("$completionLocator" -like "*:$requiredCompletionLocatorLabel") {
                    $completionLabelFound = $true
                    break
                }
            }

            if (-not $completionLabelFound) {
                $violations.Add("Completion definition matrix contract violated: $completionKey missing required locator label: $requiredCompletionLocatorLabel")
            }
        }
    }
}

$enforcementAdoptionArtifactPath = Join-Path $evidenceDir 'enforcement_adoption_report.json'
if (Test-Path -LiteralPath $enforcementAdoptionArtifactPath) {
    try {
        $enforcementAdoptionReport = Get-Content -LiteralPath $enforcementAdoptionArtifactPath -Raw -Encoding UTF8 | ConvertFrom-Json

        if ($null -eq $enforcementAdoptionReport.withinTarget -or -not [bool]$enforcementAdoptionReport.withinTarget) {
            $violations.Add('Enforcement adoption evidence requires withinTarget=true')
        }

        $advancedCoverageRatio = 0.0
        if (-not [double]::TryParse("$($enforcementAdoptionReport.advancedCoverageRatio)", [ref]$advancedCoverageRatio)) {
            $violations.Add('Enforcement adoption evidence advancedCoverageRatio parse failed')
        }

        $minAdvancedCoverageRatio = 0.0
        if (-not [double]::TryParse("$($enforcementAdoptionReport.minAdvancedCoverageRatio)", [ref]$minAdvancedCoverageRatio)) {
            $violations.Add('Enforcement adoption evidence minAdvancedCoverageRatio parse failed')
        }
        elseif ($advancedCoverageRatio -lt $minAdvancedCoverageRatio) {
            $violations.Add("Enforcement adoption evidence ratio below target: ratio=$advancedCoverageRatio target=$minAdvancedCoverageRatio")
        }

        $advancedSourceCount = [int]$enforcementAdoptionReport.advancedSourceCount
        $totalTrackedSources = [int]$enforcementAdoptionReport.totalTrackedSources
        if ($advancedSourceCount -ne $totalTrackedSources) {
            $violations.Add("Enforcement adoption evidence requires all tracked sources advanced: advanced=$advancedSourceCount total=$totalTrackedSources")
        }
    }
    catch {
        $violations.Add("Enforcement adoption evidence parse failed: path=$enforcementAdoptionArtifactPath reason=$($_.Exception.Message)")
    }
}

$enforcementSourcePurityArtifactPath = Join-Path $evidenceDir 'enforcement_source_purity_report.json'
if (Test-Path -LiteralPath $enforcementSourcePurityArtifactPath) {
    try {
        $enforcementSourcePurityReport = Get-Content -LiteralPath $enforcementSourcePurityArtifactPath -Raw -Encoding UTF8 | ConvertFrom-Json

        $rawRegexViolationCount = [int]$enforcementSourcePurityReport.rawRegexViolationCount
        $unknownViolationCount = [int]$enforcementSourcePurityReport.unknownViolationCount
        if ($rawRegexViolationCount -ne 0 -or $unknownViolationCount -ne 0) {
            $violations.Add("Enforcement source purity evidence requires zero violations: rawRegex=$rawRegexViolationCount unknown=$unknownViolationCount")
        }
    }
    catch {
        $violations.Add("Enforcement source purity evidence parse failed: path=$enforcementSourcePurityArtifactPath reason=$($_.Exception.Message)")
    }
}

$triggerAuditArtifactPath = Join-Path $evidenceDir 'trigger_audit_report.json'
if (Test-Path -LiteralPath $triggerAuditArtifactPath) {
    try {
        $triggerAuditReport = Get-Content -LiteralPath $triggerAuditArtifactPath -Raw -Encoding UTF8 | ConvertFrom-Json

        $triggerAstArtifactPath = Join-Path $evidenceDir 'trigger_ast_report.json'
        $triggerAstReport = $null
        if (-not (Test-Path -LiteralPath $triggerAstArtifactPath)) {
            $violations.Add("Trigger evidence contract violated: missing trigger AST report: path=$triggerAstArtifactPath")
        }
        else {
            try {
                $triggerAstReport = Get-Content -LiteralPath $triggerAstArtifactPath -Raw -Encoding UTF8 | ConvertFrom-Json
            }
            catch {
                $violations.Add("Trigger evidence contract violated: trigger AST report parse failed: path=$triggerAstArtifactPath reason=$($_.Exception.Message)")
            }
        }

        $triggerSymbolUsageArtifactPath = Join-Path $evidenceDir 'trigger_symbol_usage_report.json'
        $triggerSymbolUsageReport = $null
        if (Test-Path -LiteralPath $triggerSymbolUsageArtifactPath) {
            try {
                $triggerSymbolUsageReport = Get-Content -LiteralPath $triggerSymbolUsageArtifactPath -Raw -Encoding UTF8 | ConvertFrom-Json
                if ($null -eq $triggerSymbolUsageReport.policyViolations) {
                    $violations.Add('Trigger symbol usage evidence missing policyViolations field')
                }
                elseif (@($triggerSymbolUsageReport.policyViolations).Count -ne 0) {
                    $violations.Add("Trigger symbol usage evidence requires policyViolations=0 but was $(@($triggerSymbolUsageReport.policyViolations).Count)")
                }
            }
            catch {
                $violations.Add("Trigger symbol usage evidence parse failed: path=$triggerSymbolUsageArtifactPath reason=$($_.Exception.Message)")
            }
        }

        $observeShimArtifactPath = Join-Path $evidenceDir 'observe_shim_usage_report.json'
        $observeShimReport = $null
        if (Test-Path -LiteralPath $observeShimArtifactPath) {
            try {
                $observeShimReport = Get-Content -LiteralPath $observeShimArtifactPath -Raw -Encoding UTF8 | ConvertFrom-Json
                if ($null -eq $observeShimReport.policyViolations) {
                    $violations.Add('Observe shim evidence missing policyViolations field')
                }
                elseif (@($observeShimReport.policyViolations).Count -ne 0) {
                    $violations.Add("Observe shim evidence requires policyViolations=0 but was $(@($observeShimReport.policyViolations).Count)")
                }
            }
            catch {
                $violations.Add("Observe shim evidence parse failed: path=$observeShimArtifactPath reason=$($_.Exception.Message)")
            }
        }

        $requiredTriggerMetricSources = [ordered]@{
            activeDspMetricSource                   = 'symbolBlockedMatchesRefreshed'
            retireFacadeMetricSource                = 'symbolBlockedMatchesRefreshed'
            retireFacadeRuntimeExecutionMetricSource = 'symbolTotalMatchesRefreshed'
            runtimeExecutionViewMetricSource        = 'symbolBlockedMatchesRefreshed'
            fadingOutDspMetricSource                = 'triggerAstEffectiveMatches'
            legacyDirectObserveMetricSource         = 'observeShimBlockedMatches'
        }

        foreach ($requiredMetricSourceField in $requiredTriggerMetricSources.Keys) {
            if ($null -eq $triggerAuditReport.$requiredMetricSourceField) {
                $violations.Add("Trigger audit evidence missing metric source field: $requiredMetricSourceField")
                continue
            }

            $expectedMetricSourceValue = "$($requiredTriggerMetricSources[$requiredMetricSourceField])"
            $actualMetricSourceValue = "$($triggerAuditReport.$requiredMetricSourceField)"
            if ($actualMetricSourceValue -ne $expectedMetricSourceValue) {
                $violations.Add("Trigger audit evidence metric source mismatch: field=$requiredMetricSourceField expected=$expectedMetricSourceValue actual=$actualMetricSourceValue")
            }
        }

        if ($null -ne $triggerSymbolUsageReport -and $null -ne $triggerAuditReport.metrics) {
            if ($null -eq $triggerAuditReport.metrics.activeDspRawRefCount) {
                $violations.Add('Trigger audit evidence missing metrics.activeDspRawRefCount field')
            }
            elseif ([int]$triggerAuditReport.metrics.activeDspRawRefCount -ne [int]$triggerSymbolUsageReport.totalMatches) {
                $violations.Add("Trigger evidence contract violated: activeDspRawRefCount mismatch: audit=$($triggerAuditReport.metrics.activeDspRawRefCount) symbol=$($triggerSymbolUsageReport.totalMatches)")
            }

            if ($null -eq $triggerAuditReport.metrics.activeDspRefCount) {
                $violations.Add('Trigger audit evidence missing metrics.activeDspRefCount field')
            }
            elseif ([int]$triggerAuditReport.metrics.activeDspRefCount -ne [int]$triggerSymbolUsageReport.blockedMatches) {
                $violations.Add("Trigger evidence contract violated: activeDspRefCount mismatch: audit=$($triggerAuditReport.metrics.activeDspRefCount) symbolBlocked=$($triggerSymbolUsageReport.blockedMatches)")
            }

            if ($null -eq $triggerAuditReport.metrics.retireFacadeRawDependencyCount) {
                $violations.Add('Trigger audit evidence missing metrics.retireFacadeRawDependencyCount field')
            }
            elseif ([int]$triggerAuditReport.metrics.retireFacadeRawDependencyCount -ne [int]$triggerSymbolUsageReport.totalMatches) {
                $violations.Add("Trigger evidence contract violated: retireFacadeRawDependencyCount mismatch: audit=$($triggerAuditReport.metrics.retireFacadeRawDependencyCount) symbol=$($triggerSymbolUsageReport.totalMatches)")
            }

            if ($null -eq $triggerAuditReport.metrics.retireFacadeDirectDependencyCount) {
                $violations.Add('Trigger audit evidence missing metrics.retireFacadeDirectDependencyCount field')
            }
            elseif ([int]$triggerAuditReport.metrics.retireFacadeDirectDependencyCount -ne [int]$triggerSymbolUsageReport.blockedMatches) {
                $violations.Add("Trigger evidence contract violated: retireFacadeDirectDependencyCount mismatch: audit=$($triggerAuditReport.metrics.retireFacadeDirectDependencyCount) symbolBlocked=$($triggerSymbolUsageReport.blockedMatches)")
            }

            if ($null -eq $triggerAuditReport.metrics.retireFacadeRuntimeExecutionCount) {
                $violations.Add('Trigger audit evidence missing metrics.retireFacadeRuntimeExecutionCount field')
            }
            elseif ([int]$triggerAuditReport.metrics.retireFacadeRuntimeExecutionCount -ne [int]$triggerSymbolUsageReport.totalMatches) {
                $violations.Add("Trigger evidence contract violated: retireFacadeRuntimeExecutionCount mismatch: audit=$($triggerAuditReport.metrics.retireFacadeRuntimeExecutionCount) symbol=$($triggerSymbolUsageReport.totalMatches)")
            }

            if ($null -eq $triggerAuditReport.metrics.runtimeExecutionViewUsageCount) {
                $violations.Add('Trigger audit evidence missing metrics.runtimeExecutionViewUsageCount field')
            }
            elseif ([int]$triggerAuditReport.metrics.runtimeExecutionViewUsageCount -ne [int]$triggerSymbolUsageReport.blockedMatches) {
                $violations.Add("Trigger evidence contract violated: runtimeExecutionViewUsageCount mismatch: audit=$($triggerAuditReport.metrics.runtimeExecutionViewUsageCount) symbolBlocked=$($triggerSymbolUsageReport.blockedMatches)")
            }
        }

        if ($null -ne $observeShimReport -and $null -ne $triggerAuditReport.metrics) {
            if ($null -eq $triggerAuditReport.metrics.legacyDirectObserveRawCount) {
                $violations.Add('Trigger audit evidence missing metrics.legacyDirectObserveRawCount field')
            }
            elseif ([int]$triggerAuditReport.metrics.legacyDirectObserveRawCount -ne [int]$observeShimReport.totalMatches) {
                $violations.Add("Trigger evidence contract violated: legacyDirectObserveRawCount mismatch: audit=$($triggerAuditReport.metrics.legacyDirectObserveRawCount) observe=$($observeShimReport.totalMatches)")
            }

            if ($null -eq $triggerAuditReport.metrics.legacyDirectObserveUsageCount) {
                $violations.Add('Trigger audit evidence missing metrics.legacyDirectObserveUsageCount field')
            }
            elseif ([int]$triggerAuditReport.metrics.legacyDirectObserveUsageCount -ne [int]$observeShimReport.blockedMatches) {
                $violations.Add("Trigger evidence contract violated: legacyDirectObserveUsageCount mismatch: audit=$($triggerAuditReport.metrics.legacyDirectObserveUsageCount) observeBlocked=$($observeShimReport.blockedMatches)")
            }
        }

        if ($null -ne $triggerAstReport) {
            if ($null -eq $triggerAstReport.available -or -not [bool]$triggerAstReport.available) {
                $violations.Add('Trigger AST evidence requires available=true')
            }
            if ($null -eq $triggerAstReport.commandOk -or -not [bool]$triggerAstReport.commandOk) {
                $violations.Add('Trigger AST evidence requires commandOk=true')
            }
            if ($null -eq $triggerAstReport.fadingOutDspWriteEffectiveMatches) {
                $violations.Add('Trigger AST evidence missing fadingOutDspWriteEffectiveMatches field')
            }
            elseif ([int]$triggerAuditReport.metrics.fadingOutDspWriteCount -ne [int]$triggerAstReport.fadingOutDspWriteEffectiveMatches) {
                $violations.Add("Trigger evidence contract violated: fadingOutDspWriteCount mismatch: audit=$($triggerAuditReport.metrics.fadingOutDspWriteCount) ast=$($triggerAstReport.fadingOutDspWriteEffectiveMatches)")
            }

            if ($null -eq $triggerAuditReport.astEvidenceRequired) {
                $violations.Add('Trigger audit evidence missing astEvidenceRequired field')
            }
            elseif ([bool]$triggerAuditReport.astEvidenceRequired) {
                if ($null -eq $triggerAstReport.required -or -not [bool]$triggerAstReport.required) {
                    $violations.Add('Trigger evidence contract violated: astEvidenceRequired=true requires trigger_ast.required=true')
                }
                if ("$($triggerAstReport.fadingOutDspWriteEffectiveSource)" -ne 'astOnly') {
                    $violations.Add("Trigger evidence contract violated: astEvidenceRequired=true requires fadingOutDspWriteEffectiveSource=astOnly but was $($triggerAstReport.fadingOutDspWriteEffectiveSource)")
                }
            }
        }

        if ($null -eq $triggerAuditReport.policyViolations) {
            $violations.Add('Trigger audit evidence missing policyViolations field')
        }
        elseif (@($triggerAuditReport.policyViolations).Count -ne 0) {
            $violations.Add("Trigger audit evidence requires policyViolations=0 but was $(@($triggerAuditReport.policyViolations).Count)")
        }

        $requiredTriggerPolicyEvaluationIds = @(
            'activeDspDeletionStart',
            'fadingOutDspDeletionStart',
            'retireFacadeRemovalStart',
            'observeShimRemovalStart',
            'runtimeExecutionViewConvergence'
        )

        $triggerEvaluationById = @{}
        foreach ($evaluation in @($triggerAuditReport.policyEvaluations)) {
            $evaluationId = "$($evaluation.id)"
            if (-not [string]::IsNullOrWhiteSpace($evaluationId)) {
                $triggerEvaluationById[$evaluationId] = $evaluation
            }
        }

        foreach ($requiredTriggerPolicyEvaluationId in $requiredTriggerPolicyEvaluationIds) {
            if (-not $triggerEvaluationById.ContainsKey($requiredTriggerPolicyEvaluationId)) {
                $violations.Add("Trigger audit evidence missing policy evaluation id=$requiredTriggerPolicyEvaluationId")
            }
        }

        foreach ($requiredTriggerPolicyEvaluationId in $requiredTriggerPolicyEvaluationIds) {
            if (-not $triggerEvaluationById.ContainsKey($requiredTriggerPolicyEvaluationId)) {
                continue
            }

            $requiredEvaluation = $triggerEvaluationById[$requiredTriggerPolicyEvaluationId]
            if ($null -eq $requiredEvaluation.expired) {
                $violations.Add("Trigger audit evidence policy evaluation missing expired field: id=$requiredTriggerPolicyEvaluationId")
            }
            elseif ([bool]$requiredEvaluation.expired) {
                $violations.Add("Trigger audit evidence policy evaluation expired: id=$requiredTriggerPolicyEvaluationId")
            }

            $actualValue = 0.0
            if (-not [double]::TryParse("$($requiredEvaluation.actual)", [ref]$actualValue)) {
                $violations.Add("Trigger audit evidence policy evaluation actual parse failed: id=$requiredTriggerPolicyEvaluationId")
                continue
            }

            $allowedMaxValue = 0.0
            if (-not [double]::TryParse("$($requiredEvaluation.allowedMax)", [ref]$allowedMaxValue)) {
                $violations.Add("Trigger audit evidence policy evaluation allowedMax parse failed: id=$requiredTriggerPolicyEvaluationId")
                continue
            }

            if ($actualValue -gt $allowedMaxValue) {
                $violations.Add("Trigger audit evidence policy evaluation exceeds allowedMax: id=$requiredTriggerPolicyEvaluationId actual=$actualValue allowedMax=$allowedMaxValue")
            }
        }

        $requiredTriggerMetrics = @(
            'activeDspRefCount',
            'fadingOutDspWriteCount',
            'retireFacadeDirectDependencyCount',
            'retireFacadeRuntimeExecutionCount',
            'runtimeExecutionViewUsageCount',
            'legacyDirectObserveUsageCount'
        )

        if ($null -eq $triggerAuditReport.metrics) {
            $violations.Add('Trigger audit evidence missing metrics block')
        }
        else {
            foreach ($requiredTriggerMetric in $requiredTriggerMetrics) {
                if ($null -eq $triggerAuditReport.metrics.$requiredTriggerMetric) {
                    $violations.Add("Trigger audit evidence missing metric field: $requiredTriggerMetric")
                    continue
                }

                $metricValue = 0
                if (-not [int]::TryParse("$($triggerAuditReport.metrics.$requiredTriggerMetric)", [ref]$metricValue)) {
                    $violations.Add("Trigger audit evidence metric parse failed: field=$requiredTriggerMetric")
                    continue
                }

                if ($metricValue -ne 0) {
                    $violations.Add("Trigger audit evidence requires metrics.$requiredTriggerMetric=0 but was $metricValue")
                }
            }
        }

        $triggerAuditGeneratedAt = [datetime]::MinValue
        if (-not [datetime]::TryParse("$($triggerAuditReport.generatedAt)", [ref]$triggerAuditGeneratedAt)) {
            $violations.Add('Trigger audit evidence generatedAt parse failed')
        }
        else {
            $triggerAuditAgeMinutes = ((Get-Date) - $triggerAuditGeneratedAt).TotalMinutes
            if ($triggerAuditAgeMinutes -gt $artifactFreshnessWindowMinutes) {
                $violations.Add("Trigger audit evidence freshness breach: ageMinutes=$([math]::Round($triggerAuditAgeMinutes, 2)) windowMinutes=$artifactFreshnessWindowMinutes")
            }
        }
    }
    catch {
        $violations.Add("Trigger audit evidence parse failed: path=$triggerAuditArtifactPath reason=$($_.Exception.Message)")
    }
}

$validatorTieringArtifactPath = Join-Path $evidenceDir 'validator_tiering_report.json'
if (Test-Path -LiteralPath $validatorTieringArtifactPath) {
    try {
        $validatorTieringReport = Get-Content -LiteralPath $validatorTieringArtifactPath -Raw -Encoding UTF8 | ConvertFrom-Json

        if ($null -eq $validatorTieringReport.policy) {
            $violations.Add('Validator tiering evidence missing policy block')
        }
        else {
            if ("$($validatorTieringReport.policy.schema)" -ne 'isr_validator_tiering_policy_v1') {
                $violations.Add("Validator tiering evidence policy schema mismatch: expected=isr_validator_tiering_policy_v1 actual=$($validatorTieringReport.policy.schema)")
            }

            $expectedTierBindings = [ordered]@{
                smoke = 'pr'
                standard = 'nightly'
                exhaustive = 'weekly'
            }

            foreach ($tierKey in $expectedTierBindings.Keys) {
                if ($null -eq $validatorTieringReport.policy.tiers -or $null -eq $validatorTieringReport.policy.tiers.$tierKey) {
                    $violations.Add("Validator tiering evidence policy missing tiers.$tierKey")
                    continue
                }

                $actualTierBinding = "$($validatorTieringReport.policy.tiers.$tierKey)"
                $expectedTierBinding = "$($expectedTierBindings[$tierKey])"
                if ($actualTierBinding -ne $expectedTierBinding) {
                    $violations.Add("Validator tiering evidence policy tier binding mismatch: tier=$tierKey expected=$expectedTierBinding actual=$actualTierBinding")
                }
            }

            if ($null -eq $validatorTieringReport.policy.slaHours -or $null -eq $validatorTieringReport.policy.slaHours.hbViolation) {
                $violations.Add('Validator tiering evidence policy missing slaHours.hbViolation')
            }
            elseif ([int]$validatorTieringReport.policy.slaHours.hbViolation -ne 24) {
                $violations.Add("Validator tiering evidence policy slaHours.hbViolation mismatch: expected=24 actual=$($validatorTieringReport.policy.slaHours.hbViolation)")
            }

            if ($null -eq $validatorTieringReport.policy.slaHours -or $null -eq $validatorTieringReport.policy.slaHours.payloadMismatch) {
                $violations.Add('Validator tiering evidence policy missing slaHours.payloadMismatch')
            }
            elseif ([int]$validatorTieringReport.policy.slaHours.payloadMismatch -ne 72) {
                $violations.Add("Validator tiering evidence policy slaHours.payloadMismatch mismatch: expected=72 actual=$($validatorTieringReport.policy.slaHours.payloadMismatch)")
            }
        }

        if ($null -eq $validatorTieringReport.slaFreshness) {
            $violations.Add('Validator tiering evidence missing slaFreshness block')
        }
        else {
            $slaByKey = @{}
            foreach ($slaEntry in @($validatorTieringReport.slaFreshness)) {
                $slaKey = "$($slaEntry.key)"
                if (-not [string]::IsNullOrWhiteSpace($slaKey)) {
                    $slaByKey[$slaKey] = $slaEntry
                }
            }

            $requiredSlaKeys = [ordered]@{
                hbViolation = 24
                payloadMismatch = 72
            }

            foreach ($requiredSlaKey in $requiredSlaKeys.Keys) {
                if (-not $slaByKey.ContainsKey($requiredSlaKey)) {
                    $violations.Add("Validator tiering evidence missing slaFreshness entry: key=$requiredSlaKey")
                    continue
                }

                $slaEntry = $slaByKey[$requiredSlaKey]
                $expectedMaxAgeHours = [int]$requiredSlaKeys[$requiredSlaKey]

                if ($null -eq $slaEntry.present -or -not [bool]$slaEntry.present) {
                    $violations.Add("Validator tiering evidence requires slaFreshness.present=true: key=$requiredSlaKey")
                }

                if ($null -eq $slaEntry.withinSla -or -not [bool]$slaEntry.withinSla) {
                    $violations.Add("Validator tiering evidence requires slaFreshness.withinSla=true: key=$requiredSlaKey")
                }

                if ($null -eq $slaEntry.maxAgeHours) {
                    $violations.Add("Validator tiering evidence missing slaFreshness.maxAgeHours: key=$requiredSlaKey")
                }
                elseif ([int]$slaEntry.maxAgeHours -ne $expectedMaxAgeHours) {
                    $violations.Add("Validator tiering evidence slaFreshness.maxAgeHours mismatch: key=$requiredSlaKey expected=$expectedMaxAgeHours actual=$($slaEntry.maxAgeHours)")
                }
            }
        }

        if ($null -eq $validatorTieringReport.violations) {
            $violations.Add('Validator tiering evidence missing violations field')
        }
        elseif (@($validatorTieringReport.violations).Count -ne 0) {
            $violations.Add("Validator tiering evidence requires violations=0 but was $(@($validatorTieringReport.violations).Count)")
        }
    }
    catch {
        $violations.Add("Validator tiering evidence parse failed: path=$validatorTieringArtifactPath reason=$($_.Exception.Message)")
    }
}

foreach ($freshnessArtifact in @(
    @{ Path = (Join-Path $evidenceDir 'trigger_symbol_usage_report.json'); Label = 'Trigger symbol usage evidence' },
    @{ Path = (Join-Path $evidenceDir 'observe_shim_usage_report.json'); Label = 'Observe shim evidence' },
    @{ Path = (Join-Path $evidenceDir 'trigger_ast_report.json'); Label = 'Trigger AST evidence' },
    @{ Path = (Join-Path $evidenceDir 'validator_tiering_report.json'); Label = 'Validator tiering evidence' },
        @{ Path = (Join-Path $evidenceDir 'trigger_audit_report.json'); Label = 'Trigger audit evidence' },
        @{ Path = (Join-Path $evidenceDir 'metric_governance_report.json'); Label = 'Metric governance evidence' },
        @{ Path = (Join-Path $evidenceDir 'flag_dependency_graph_report.json'); Label = 'Flag dependency evidence' },
        @{ Path = (Join-Path $evidenceDir 'enforcement_adoption_report.json'); Label = 'Enforcement adoption evidence' },
        @{ Path = (Join-Path $evidenceDir 'enforcement_source_purity_report.json'); Label = 'Enforcement source purity evidence' }
    )) {
    if (Test-Path -LiteralPath $freshnessArtifact.Path) {
        try {
            $freshnessReport = Get-Content -LiteralPath $freshnessArtifact.Path -Raw -Encoding UTF8 | ConvertFrom-Json
            $freshnessGeneratedAt = [datetime]::MinValue
            if (-not [datetime]::TryParse("$($freshnessReport.generatedAt)", [ref]$freshnessGeneratedAt)) {
                $violations.Add("$($freshnessArtifact.Label) generatedAt parse failed")
            }
            else {
                $freshnessAgeMinutes = ((Get-Date) - $freshnessGeneratedAt).TotalMinutes
                if ($freshnessAgeMinutes -gt $artifactFreshnessWindowMinutes) {
                    $violations.Add("$($freshnessArtifact.Label) freshness breach: ageMinutes=$([math]::Round($freshnessAgeMinutes, 2)) windowMinutes=$artifactFreshnessWindowMinutes")
                }
            }
        }
        catch {
            $violations.Add("$($freshnessArtifact.Label) freshness parse failed: path=$($freshnessArtifact.Path) reason=$($_.Exception.Message)")
        }
    }
}

$canaryNormalizationArtifactPath = Join-Path $evidenceDir 'canary_baseline_normalization_report.json'
if (Test-Path -LiteralPath $canaryNormalizationArtifactPath) {
    try {
        $canaryNormalizationReport = Get-Content -LiteralPath $canaryNormalizationArtifactPath -Raw -Encoding UTF8 | ConvertFrom-Json
        $canaryMetricIds = @($canaryNormalizationReport.metrics | ForEach-Object { "$($_.id)" })

        foreach ($requiredCanaryMetricId in @('xrunDelta', 'callbackJitter', 'retireLatency', 'crossfadePeak')) {
            if ($canaryMetricIds -notcontains $requiredCanaryMetricId) {
                $violations.Add("Canary normalization evidence missing required metric id=$requiredCanaryMetricId")
            }
        }

        foreach ($metric in @($canaryNormalizationReport.metrics)) {
            if ($null -eq $metric.normalizationApplied -or -not [bool]$metric.normalizationApplied) {
                $violations.Add("Canary normalization evidence requires normalizationApplied=true for metric id=$($metric.id)")
            }
        }

        if ($null -eq $canaryNormalizationReport.violations) {
            $violations.Add('Canary normalization evidence missing violations field')
        }
        elseif (@($canaryNormalizationReport.violations).Count -ne 0) {
            $violations.Add("Canary normalization evidence requires violations=0 but was $(@($canaryNormalizationReport.violations).Count)")
        }

        $canaryGeneratedAt = [datetime]::MinValue
        if (-not [datetime]::TryParse("$($canaryNormalizationReport.generatedAt)", [ref]$canaryGeneratedAt)) {
            $violations.Add('Canary normalization evidence generatedAt parse failed')
        }
        else {
            $canaryAgeMinutes = ((Get-Date) - $canaryGeneratedAt).TotalMinutes
            if ($canaryAgeMinutes -gt $artifactFreshnessWindowMinutes) {
                $violations.Add("Canary normalization evidence freshness breach: ageMinutes=$([math]::Round($canaryAgeMinutes, 2)) windowMinutes=$artifactFreshnessWindowMinutes")
            }
        }
    }
    catch {
        $violations.Add("Canary normalization evidence parse failed: path=$canaryNormalizationArtifactPath reason=$($_.Exception.Message)")
    }
}

$metricGovernanceArtifactPath = Join-Path $evidenceDir 'metric_governance_report.json'
if (Test-Path -LiteralPath $metricGovernanceArtifactPath) {
    try {
        $metricGovernanceReport = Get-Content -LiteralPath $metricGovernanceArtifactPath -Raw -Encoding UTF8 | ConvertFrom-Json
        $metricNormalizationPolicy = $metricGovernanceReport.normalizationPolicy

        if ([string]::IsNullOrWhiteSpace("$($metricGovernanceReport.registryOwner)")) {
            $violations.Add('Metric governance evidence missing registryOwner field')
        }

        if ([string]::IsNullOrWhiteSpace("$($metricGovernanceReport.registryIssue)")) {
            $violations.Add('Metric governance evidence missing registryIssue field')
        }

        $metricRegistryExpiry = [datetime]::MinValue
        if ([string]::IsNullOrWhiteSpace("$($metricGovernanceReport.registryExpiry)")) {
            $violations.Add('Metric governance evidence missing registryExpiry field')
        }
        elseif (-not [datetime]::TryParse("$($metricGovernanceReport.registryExpiry)", [ref]$metricRegistryExpiry)) {
            $violations.Add("Metric governance evidence registryExpiry parse failed: value=$($metricGovernanceReport.registryExpiry)")
        }
        elseif ((Get-Date) -gt $metricRegistryExpiry) {
            $violations.Add("Metric governance evidence registryExpiry expired: value=$($metricGovernanceReport.registryExpiry)")
        }

        if ($null -eq $metricNormalizationPolicy -or -not [bool]$metricNormalizationPolicy.enabled) {
            $violations.Add('Metric governance evidence requires normalizationPolicy.enabled=true')
        }

        if ($null -eq $metricNormalizationPolicy -or -not [bool]$metricNormalizationPolicy.strictModeRequireAllMetrics) {
            $violations.Add('Metric governance evidence requires normalizationPolicy.strictModeRequireAllMetrics=true')
        }

        $baselineWindowMinutes = -1
        if ($null -eq $metricNormalizationPolicy -or -not [int]::TryParse("$($metricNormalizationPolicy.baselineWindowMinutes)", [ref]$baselineWindowMinutes)) {
            $violations.Add('Metric governance evidence baselineWindowMinutes parse failed')
        }
        elseif ($baselineWindowMinutes -le 0 -or $baselineWindowMinutes -gt 60) {
            $violations.Add("Metric governance evidence baselineWindowMinutes out of range: value=$baselineWindowMinutes expected=1..60")
        }

        $governedMetricIds = @($metricGovernanceReport.metrics | ForEach-Object { "$($_.id)" })
        foreach ($requiredGovernedMetricId in @('xrunDelta', 'callbackJitter', 'retireLatency', 'crossfadePeak')) {
            if ($governedMetricIds -notcontains $requiredGovernedMetricId) {
                $violations.Add("Metric governance evidence missing required metric id=$requiredGovernedMetricId")
            }
        }

        $governedMetricCount = @($governedMetricIds).Count
        if ($governedMetricCount -gt 4) {
            $violations.Add("Metric governance evidence exceeds controlled canary metric set: metricCount=$governedMetricCount max=4")
        }

        foreach ($metric in @($metricGovernanceReport.metrics)) {
            $metricId = "$($metric.id)"

            if ([string]::IsNullOrWhiteSpace($metricId)) {
                $violations.Add('Metric governance evidence metric entry missing id')
                continue
            }

            if ($null -eq $metric.blocking -or ("$($metric.blocking)" -ne 'yes' -and "$($metric.blocking)" -ne 'no')) {
                $violations.Add("Metric governance evidence metric blocking must be yes/no: id=$metricId actual=$($metric.blocking)")
            }

            if ([string]::IsNullOrWhiteSpace("$($metric.owner)")) {
                $violations.Add("Metric governance evidence metric missing owner: id=$metricId")
            }

            if ([string]::IsNullOrWhiteSpace("$($metric.retention)")) {
                $violations.Add("Metric governance evidence metric missing retention: id=$metricId")
            }

            if ([string]::IsNullOrWhiteSpace("$($metric.threshold)")) {
                $violations.Add("Metric governance evidence metric missing threshold: id=$metricId")
            }

            if ([string]::IsNullOrWhiteSpace("$($metric.action)")) {
                $violations.Add("Metric governance evidence metric missing action: id=$metricId")
            }

            if ("$($metric.normalization)" -ne 'baselineWindowNormalized') {
                $violations.Add("Metric governance evidence metric normalization mismatch: id=$metricId expected=baselineWindowNormalized actual=$($metric.normalization)")
            }

            if ([string]::IsNullOrWhiteSpace("$($metric.issue)")) {
                $violations.Add("Metric governance evidence metric missing issue: id=$metricId")
            }

            if ($null -eq $metric.expired) {
                $violations.Add("Metric governance evidence metric missing expired field: id=$metricId")
            }
            elseif ([bool]$metric.expired) {
                $violations.Add("Metric governance evidence metric expired: id=$metricId")
            }
        }

        if ($null -eq $metricGovernanceReport.violations) {
            $violations.Add('Metric governance evidence missing violations field')
        }
        elseif (@($metricGovernanceReport.violations).Count -ne 0) {
            $violations.Add("Metric governance evidence requires violations=0 but was $(@($metricGovernanceReport.violations).Count)")
        }
    }
    catch {
        $violations.Add("Metric governance evidence parse failed: path=$metricGovernanceArtifactPath reason=$($_.Exception.Message)")
    }
}

$flagDependencyArtifactPath = Join-Path $evidenceDir 'flag_dependency_graph_report.json'
$rollbackMatrixPolicyPath = Join-Path $repoRoot '.github\isr-rollback-compatibility-matrix.json'
if ((Test-Path -LiteralPath $flagDependencyArtifactPath) -and (Test-Path -LiteralPath $rollbackMatrixPolicyPath)) {
    try {
        $flagDependencyReport = Get-Content -LiteralPath $flagDependencyArtifactPath -Raw -Encoding UTF8 | ConvertFrom-Json
        $rollbackMatrixPolicy = Get-Content -LiteralPath $rollbackMatrixPolicyPath -Raw -Encoding UTF8 | ConvertFrom-Json

        if ($null -eq $flagDependencyReport.hasCycle) {
            $violations.Add('Flag dependency evidence missing hasCycle field')
        }
        elseif ([bool]$flagDependencyReport.hasCycle) {
            $violations.Add('Flag dependency evidence requires hasCycle=false')
        }

        $expectedNodeCount = 1 + @($rollbackMatrixPolicy.subsystemFlags).Count
        $actualNodeCount = [int]$flagDependencyReport.nodeCount
        if ($actualNodeCount -ne $expectedNodeCount) {
            $violations.Add("Flag dependency evidence node count mismatch with rollback matrix: actual=$actualNodeCount expected=$expectedNodeCount")
        }

        $actualGlobalFlag = "$($flagDependencyReport.globalFlag)"
        $expectedGlobalFlag = "$($rollbackMatrixPolicy.globalFlag)"
        if ([string]::IsNullOrWhiteSpace($actualGlobalFlag) -or $actualGlobalFlag -ne $expectedGlobalFlag) {
            $violations.Add("Flag dependency evidence global flag mismatch: actual=$actualGlobalFlag expected=$expectedGlobalFlag")
        }
    }
    catch {
        $violations.Add("Flag dependency evidence parse failed: path=$flagDependencyArtifactPath reason=$($_.Exception.Message)")
    }
}

$rollbackCompatibilityArtifactPath = Join-Path $evidenceDir 'rollback_compatibility_report.json'
if ((Test-Path -LiteralPath $rollbackCompatibilityArtifactPath) -and (Test-Path -LiteralPath $rollbackMatrixPolicyPath)) {
    try {
        $rollbackCompatibilityReport = Get-Content -LiteralPath $rollbackCompatibilityArtifactPath -Raw -Encoding UTF8 | ConvertFrom-Json
        $rollbackMatrixPolicy = Get-Content -LiteralPath $rollbackMatrixPolicyPath -Raw -Encoding UTF8 | ConvertFrom-Json

        $actualMatrixPath = "$($rollbackCompatibilityReport.matrixPath)"
        if ([string]::IsNullOrWhiteSpace($actualMatrixPath) -or [System.IO.Path]::GetFullPath($actualMatrixPath) -ne $rollbackMatrixPolicyPath) {
            $violations.Add("Rollback compatibility evidence matrixPath mismatch: expected=$rollbackMatrixPolicyPath actual=$actualMatrixPath")
        }

        if ([string]::IsNullOrWhiteSpace("$($rollbackCompatibilityReport.matrixOwner)")) {
            $violations.Add('Rollback compatibility evidence missing matrixOwner field')
        }

        if ([string]::IsNullOrWhiteSpace("$($rollbackCompatibilityReport.matrixIssue)")) {
            $violations.Add('Rollback compatibility evidence missing matrixIssue field')
        }

        $matrixExpiry = [datetime]::MinValue
        if ([string]::IsNullOrWhiteSpace("$($rollbackCompatibilityReport.matrixExpiry)")) {
            $violations.Add('Rollback compatibility evidence missing matrixExpiry field')
        }
        elseif (-not [datetime]::TryParse("$($rollbackCompatibilityReport.matrixExpiry)", [ref]$matrixExpiry)) {
            $violations.Add("Rollback compatibility evidence matrixExpiry parse failed: value=$($rollbackCompatibilityReport.matrixExpiry)")
        }
        elseif ((Get-Date) -gt $matrixExpiry) {
            $violations.Add("Rollback compatibility evidence matrixExpiry expired: value=$($rollbackCompatibilityReport.matrixExpiry)")
        }

        if ("$($rollbackCompatibilityReport.globalFlag)" -ne "$($rollbackMatrixPolicy.globalFlag)") {
            $violations.Add("Rollback compatibility evidence globalFlag mismatch: expected=$($rollbackMatrixPolicy.globalFlag) actual=$($rollbackCompatibilityReport.globalFlag)")
        }

        $expectedSubsystemFlagCount = @($rollbackMatrixPolicy.subsystemFlags).Count
        if ($null -eq $rollbackCompatibilityReport.subsystemFlagCount) {
            $violations.Add('Rollback compatibility evidence missing subsystemFlagCount field')
        }
        elseif ([int]$rollbackCompatibilityReport.subsystemFlagCount -ne $expectedSubsystemFlagCount) {
            $violations.Add("Rollback compatibility evidence subsystemFlagCount mismatch: expected=$expectedSubsystemFlagCount actual=$($rollbackCompatibilityReport.subsystemFlagCount)")
        }

        $expectedScenarioCount = @($rollbackMatrixPolicy.compatibility).Count
        if ($null -eq $rollbackCompatibilityReport.scenarioCount) {
            $violations.Add('Rollback compatibility evidence missing scenarioCount field')
        }
        elseif ([int]$rollbackCompatibilityReport.scenarioCount -ne $expectedScenarioCount) {
            $violations.Add("Rollback compatibility evidence scenarioCount mismatch: expected=$expectedScenarioCount actual=$($rollbackCompatibilityReport.scenarioCount)")
        }

        $coverage = $rollbackCompatibilityReport.metricActionCoverage
        foreach ($subsystemFlag in @($rollbackMatrixPolicy.subsystemFlags)) {
            $subsystemFlagName = "$($subsystemFlag.flag)"
            if ($null -eq $coverage.$subsystemFlagName) {
                $violations.Add("Rollback compatibility evidence missing metricActionCoverage for flag=$subsystemFlagName")
                continue
            }

            $coverageValue = -1
            if (-not [int]::TryParse("$($coverage.$subsystemFlagName)", [ref]$coverageValue)) {
                $violations.Add("Rollback compatibility evidence metricActionCoverage parse failed: flag=$subsystemFlagName")
                continue
            }

            if ($coverageValue -le 0) {
                $violations.Add("Rollback compatibility evidence requires metricActionCoverage>0: flag=$subsystemFlagName actual=$coverageValue")
            }
        }

        $scenarioByName = @{}
        foreach ($scenario in @($rollbackCompatibilityReport.scenarios)) {
            $scenarioName = "$($scenario.scenario)"
            if (-not [string]::IsNullOrWhiteSpace($scenarioName)) {
                $scenarioByName[$scenarioName] = $scenario
            }
        }

        foreach ($expectedScenario in @($rollbackMatrixPolicy.compatibility)) {
            $expectedScenarioName = "$($expectedScenario.scenario)"
            if (-not $scenarioByName.ContainsKey($expectedScenarioName)) {
                $violations.Add("Rollback compatibility evidence missing scenario: scenario=$expectedScenarioName")
                continue
            }

            $actualScenario = $scenarioByName[$expectedScenarioName]
            $expectedRequiredFlags = @($expectedScenario.requiredFlags | ForEach-Object { "$_" }) | Sort-Object
            $actualRequiredFlags = @($actualScenario.requiredFlags | ForEach-Object { "$_" }) | Sort-Object
            if (($expectedRequiredFlags.Count -ne $actualRequiredFlags.Count) -or (-not (@($expectedRequiredFlags) -join '|').Equals((@($actualRequiredFlags) -join '|'), [System.StringComparison]::Ordinal))) {
                $violations.Add("Rollback compatibility evidence requiredFlags mismatch: scenario=$expectedScenarioName expected=$(@($expectedRequiredFlags) -join ',') actual=$(@($actualRequiredFlags) -join ',')")
            }
        }

        if ($null -eq $rollbackCompatibilityReport.violations) {
            $violations.Add('Rollback compatibility evidence missing violations field')
        }
        elseif (@($rollbackCompatibilityReport.violations).Count -ne 0) {
            $violations.Add("Rollback compatibility evidence requires violations=0 but was $(@($rollbackCompatibilityReport.violations).Count)")
        }
    }
    catch {
        $violations.Add("Rollback compatibility evidence parse failed: path=$rollbackCompatibilityArtifactPath reason=$($_.Exception.Message)")
    }
}

$triggerCleanupCompletionArtifactPath = Join-Path $evidenceDir 'trigger_cleanup_completion_report.json'
if (Test-Path -LiteralPath $triggerCleanupCompletionArtifactPath) {
    try {
        $triggerCleanupCompletion = Get-Content -LiteralPath $triggerCleanupCompletionArtifactPath -Raw -Encoding UTF8 | ConvertFrom-Json
        $actualDeferredRegistryPath = "$($triggerCleanupCompletion.deferredRegistryPath)"
        $actualSourceRoot = "$($triggerCleanupCompletion.sourceRoot)"

        if ($null -eq $triggerCleanupCompletion.cleanupCompleted -or -not [bool]$triggerCleanupCompletion.cleanupCompleted) {
            $violations.Add('Trigger cleanup completion evidence requires cleanupCompleted=true')
        }

        if ($null -eq $triggerCleanupCompletion.deferredRegistryEntryCount -or [int]$triggerCleanupCompletion.deferredRegistryEntryCount -ne 0) {
            $violations.Add("Trigger cleanup completion evidence requires deferredRegistryEntryCount=0 but was $($triggerCleanupCompletion.deferredRegistryEntryCount)")
        }

        if ([string]::IsNullOrWhiteSpace($actualDeferredRegistryPath) -or [System.IO.Path]::GetFullPath($actualDeferredRegistryPath) -ne $expectedDeferredRegistryPath) {
            $violations.Add("Trigger cleanup completion evidence deferredRegistryPath mismatch: expected=$expectedDeferredRegistryPath actual=$($triggerCleanupCompletion.deferredRegistryPath)")
        }
        else {
            $deferredRegistryPathSatisfied = $true
        }

        if ([string]::IsNullOrWhiteSpace($actualSourceRoot) -or [System.IO.Path]::GetFullPath($actualSourceRoot) -ne $expectedSourceRoot) {
            $violations.Add("Trigger cleanup completion evidence sourceRoot mismatch: expected=$expectedSourceRoot actual=$($triggerCleanupCompletion.sourceRoot)")
        }
        else {
            $sourceRootSatisfied = $true
        }

        $triggerCleanupGeneratedAt = [datetime]::MinValue
        if (-not [datetime]::TryParse("$($triggerCleanupCompletion.generatedAt)", [ref]$triggerCleanupGeneratedAt)) {
            $violations.Add('Trigger cleanup completion evidence generatedAt parse failed')
        }
        else {
            $triggerCleanupAgeMinutes = ((Get-Date) - $triggerCleanupGeneratedAt).TotalMinutes
            if ($triggerCleanupAgeMinutes -gt $artifactFreshnessWindowMinutes) {
                $violations.Add("Trigger cleanup completion evidence freshness breach: ageMinutes=$([math]::Round($triggerCleanupAgeMinutes, 2)) windowMinutes=$artifactFreshnessWindowMinutes")
            }
        }
    }
    catch {
        $violations.Add("Trigger cleanup completion evidence parse failed: path=$triggerCleanupCompletionArtifactPath reason=$($_.Exception.Message)")
    }
}

$triggerCleanupReadinessArtifactPath = Join-Path $evidenceDir 'trigger_cleanup_readiness_report.json'
if (Test-Path -LiteralPath $triggerCleanupReadinessArtifactPath) {
    try {
        $triggerCleanupReadiness = Get-Content -LiteralPath $triggerCleanupReadinessArtifactPath -Raw -Encoding UTF8 | ConvertFrom-Json

        if ($null -eq $triggerCleanupReadiness.readyCount -or [int]$triggerCleanupReadiness.readyCount -ne 0) {
            $violations.Add("Trigger cleanup readiness evidence requires readyCount=0 but was $($triggerCleanupReadiness.readyCount)")
        }

        if ($null -eq $triggerCleanupReadiness.blockedCount -or [int]$triggerCleanupReadiness.blockedCount -ne 0) {
            $violations.Add("Trigger cleanup readiness evidence requires blockedCount=0 but was $($triggerCleanupReadiness.blockedCount)")
        }

        $readinessGeneratedAt = [datetime]::MinValue
        if (-not [datetime]::TryParse("$($triggerCleanupReadiness.generatedAt)", [ref]$readinessGeneratedAt)) {
            $violations.Add('Trigger cleanup readiness evidence generatedAt parse failed')
        }
        else {
            $readinessAgeMinutes = ((Get-Date) - $readinessGeneratedAt).TotalMinutes
            if ($readinessAgeMinutes -gt $artifactFreshnessWindowMinutes) {
                $violations.Add("Trigger cleanup readiness evidence freshness breach: ageMinutes=$([math]::Round($readinessAgeMinutes, 2)) windowMinutes=$artifactFreshnessWindowMinutes")
            }
        }
    }
    catch {
        $violations.Add("Trigger cleanup readiness evidence parse failed: path=$triggerCleanupReadinessArtifactPath reason=$($_.Exception.Message)")
    }
}

$ownershipMigrationArtifactPath = Join-Path $evidenceDir 'ownership_migration_report.json'
if (Test-Path -LiteralPath $ownershipMigrationArtifactPath) {
    try {
        $ownershipMigrationReport = Get-Content -LiteralPath $ownershipMigrationArtifactPath -Raw -Encoding UTF8 | ConvertFrom-Json

        $ownershipTriggerAuditReportPath = "$($ownershipMigrationReport.triggerAuditReportPath)"
        if ([string]::IsNullOrWhiteSpace($ownershipTriggerAuditReportPath) -or [System.IO.Path]::GetFullPath($ownershipTriggerAuditReportPath) -ne $triggerAuditArtifactPath) {
            $violations.Add("Ownership migration evidence triggerAuditReportPath mismatch: expected=$triggerAuditArtifactPath actual=$($ownershipMigrationReport.triggerAuditReportPath)")
        }

        if ($null -eq $ownershipMigrationReport.violations) {
            $violations.Add('Ownership migration evidence missing violations field')
        }
        elseif (@($ownershipMigrationReport.violations).Count -ne 0) {
            $violations.Add("Ownership migration evidence requires violations=0 but was $(@($ownershipMigrationReport.violations).Count)")
        }

        if ($null -eq $ownershipMigrationReport.authorityTransferSequence) {
            $violations.Add('Ownership migration evidence missing authorityTransferSequence field')
        }
        else {
            $authorityTransferSequenceAllTrue = $true
            foreach ($authorityTransferStepField in @(
                    'newAuthorityIntroduced',
                    'readPathCutoverVerified',
                    'writePathCutoverVerified',
                    'metricsConfirmed',
                    'triggerConfirmed',
                    'legacyAuthorityRemoved'
                )) {
                if ($null -eq $ownershipMigrationReport.authorityTransferSequence.$authorityTransferStepField) {
                    $violations.Add("Ownership migration evidence missing authorityTransferSequence.$authorityTransferStepField field")
                    $authorityTransferSequenceAllTrue = $false
                    continue
                }

                if (-not [bool]$ownershipMigrationReport.authorityTransferSequence.$authorityTransferStepField) {
                    $violations.Add("Ownership migration evidence requires authorityTransferSequence.$authorityTransferStepField=true")
                    $authorityTransferSequenceAllTrue = $false
                }
            }

            if ($null -ne $ownershipMigrationReport.allStepsSatisfied -and
                [bool]$ownershipMigrationReport.allStepsSatisfied -ne $authorityTransferSequenceAllTrue) {
                $violations.Add("Ownership migration evidence allStepsSatisfied mismatch: expected=$authorityTransferSequenceAllTrue actual=$($ownershipMigrationReport.allStepsSatisfied)")
            }
        }

        if ($null -eq $ownershipMigrationReport.stepDiagnostics) {
            $violations.Add('Ownership migration evidence missing stepDiagnostics field')
        }
        else {
            $requiredOwnershipSteps = @(
                'newAuthorityIntroduced',
                'readPathCutoverVerified',
                'writePathCutoverVerified',
                'metricsConfirmed',
                'triggerConfirmed',
                'legacyAuthorityRemoved'
            )

            $requiredStepLocatorLabels = @{
                newAuthorityIntroduced = @('prepare_owner_signature', 'set_rcu_provider_owner')
                readPathCutoverVerified = @('legacy_publish_forwarder', 'legacy_enter_forwarder', 'legacy_exit_forwarder')
                writePathCutoverVerified = @('prepare_pass_owner', 'legacy_owner_assignment')
                metricsConfirmed = @('metric_active_dsp_ref', 'metric_fading_write', 'metric_runtime_view')
                triggerConfirmed = @('policy_violations', 'policy_runtime_view', 'policy_fading_write')
                legacyAuthorityRemoved = @('legacy_owner_member', 'legacy_owner_symbol')
            }

            $ownershipHeaderPath = "$($ownershipMigrationReport.headerPath)"
            $ownershipLifecyclePath = "$($ownershipMigrationReport.lifecyclePath)"

            $allowedOwnershipLocatorPrefixes = @(
                "${ownershipHeaderPath}:",
                "${ownershipLifecyclePath}:",
                "${triggerAuditArtifactPath}:"
            )

            $diagnosticByStep = @{}
            foreach ($stepDiagnostic in @($ownershipMigrationReport.stepDiagnostics)) {
                $stepName = "$($stepDiagnostic.step)"
                if (-not [string]::IsNullOrWhiteSpace($stepName)) {
                    $diagnosticByStep[$stepName] = $stepDiagnostic
                }
            }

            foreach ($requiredOwnershipStep in $requiredOwnershipSteps) {
                if (-not $diagnosticByStep.ContainsKey($requiredOwnershipStep)) {
                    $violations.Add("Ownership migration evidence missing stepDiagnostics entry for step=$requiredOwnershipStep")
                    continue
                }

                $stepDiagnostic = $diagnosticByStep[$requiredOwnershipStep]

                if ($null -eq $stepDiagnostic.reason -or [string]::IsNullOrWhiteSpace("$($stepDiagnostic.reason)")) {
                    $violations.Add("Ownership migration evidence stepDiagnostics.$requiredOwnershipStep missing non-empty reason")
                }

                if ($null -eq $stepDiagnostic.evidenceLocators -or @($stepDiagnostic.evidenceLocators).Count -eq 0) {
                    $violations.Add("Ownership migration evidence stepDiagnostics.$requiredOwnershipStep missing evidenceLocators")
                }
                else {
                    foreach ($evidenceLocator in @($stepDiagnostic.evidenceLocators)) {
                        if ([string]::IsNullOrWhiteSpace("$evidenceLocator")) {
                            $violations.Add("Ownership migration evidence stepDiagnostics.$requiredOwnershipStep contains empty evidence locator")
                            continue
                        }

                        if ("$evidenceLocator" -notmatch '^.+:.+:.+$') {
                            $violations.Add("Ownership migration evidence stepDiagnostics.$requiredOwnershipStep evidence locator format invalid: locator=$evidenceLocator")
                        }

                        $prefixSatisfied = $false
                        foreach ($allowedOwnershipLocatorPrefix in $allowedOwnershipLocatorPrefixes) {
                            if ("$evidenceLocator".StartsWith($allowedOwnershipLocatorPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
                                $prefixSatisfied = $true
                                break
                            }
                        }

                        if (-not $prefixSatisfied) {
                            $violations.Add("Ownership migration evidence stepDiagnostics.$requiredOwnershipStep evidence locator prefix invalid: locator=$evidenceLocator")
                        }
                    }

                    if ($requiredStepLocatorLabels.ContainsKey($requiredOwnershipStep)) {
                        foreach ($requiredLocatorLabel in @($requiredStepLocatorLabels[$requiredOwnershipStep])) {
                            $labelFound = $false
                            foreach ($evidenceLocator in @($stepDiagnostic.evidenceLocators)) {
                                if ("$evidenceLocator" -like "*:$requiredLocatorLabel") {
                                    $labelFound = $true
                                    break
                                }
                            }

                            if (-not $labelFound) {
                                $violations.Add("Ownership migration evidence stepDiagnostics.$requiredOwnershipStep missing required locator label: $requiredLocatorLabel")
                            }
                        }
                    }
                }

                if ($null -eq $stepDiagnostic.satisfied) {
                    $violations.Add("Ownership migration evidence stepDiagnostics.$requiredOwnershipStep missing satisfied field")
                }
                elseif ([bool]$stepDiagnostic.satisfied -ne [bool]$ownershipMigrationReport.authorityTransferSequence.$requiredOwnershipStep) {
                    $violations.Add("Ownership migration evidence stepDiagnostics.$requiredOwnershipStep satisfied mismatch with authorityTransferSequence")
                }
            }
        }

        if ($null -eq $ownershipMigrationReport.allStepsSatisfied) {
            $violations.Add('Ownership migration evidence missing allStepsSatisfied field')
        }
        elseif (-not [bool]$ownershipMigrationReport.allStepsSatisfied) {
            $violations.Add('Ownership migration evidence requires allStepsSatisfied=true')
        }

        $ownershipGeneratedAt = [datetime]::MinValue
        if (-not [datetime]::TryParse("$($ownershipMigrationReport.generatedAt)", [ref]$ownershipGeneratedAt)) {
            $violations.Add('Ownership migration evidence generatedAt parse failed')
        }
        else {
            $ownershipAgeMinutes = ((Get-Date) - $ownershipGeneratedAt).TotalMinutes
            if ($ownershipAgeMinutes -gt $artifactFreshnessWindowMinutes) {
                $violations.Add("Ownership migration evidence freshness breach: ageMinutes=$([math]::Round($ownershipAgeMinutes, 2)) windowMinutes=$artifactFreshnessWindowMinutes")
            }
        }
    }
    catch {
        $violations.Add("Ownership migration evidence parse failed: path=$ownershipMigrationArtifactPath reason=$($_.Exception.Message)")
    }
}

if ((Test-Path -LiteralPath $triggerCleanupReadinessArtifactPath) -and (Test-Path -LiteralPath $triggerCleanupCompletionArtifactPath)) {
    try {
        $readinessForOrder = Get-Content -LiteralPath $triggerCleanupReadinessArtifactPath -Raw -Encoding UTF8 | ConvertFrom-Json
        $completionForOrder = Get-Content -LiteralPath $triggerCleanupCompletionArtifactPath -Raw -Encoding UTF8 | ConvertFrom-Json
        $readinessGeneratedAtOrder = [datetime]::MinValue
        $completionGeneratedAtOrder = [datetime]::MinValue

        if ([datetime]::TryParse("$($readinessForOrder.generatedAt)", [ref]$readinessGeneratedAtOrder) -and
            [datetime]::TryParse("$($completionForOrder.generatedAt)", [ref]$completionGeneratedAtOrder)) {
            if ($readinessGeneratedAtOrder -gt $completionGeneratedAtOrder) {
                $violations.Add('Cleanup sequence contract violated: readiness evidence must be generated before or at cleanup completion evidence')
            }
        }
    }
    catch {
        $violations.Add("Cleanup sequence evidence order check failed: reason=$($_.Exception.Message)")
    }
}

if ((Test-Path -LiteralPath $triggerAuditArtifactPath) -and (Test-Path -LiteralPath $triggerCleanupReadinessArtifactPath)) {
    try {
        $triggerAuditForReadinessOrder = Get-Content -LiteralPath $triggerAuditArtifactPath -Raw -Encoding UTF8 | ConvertFrom-Json
        $readinessForTriggerOrder = Get-Content -LiteralPath $triggerCleanupReadinessArtifactPath -Raw -Encoding UTF8 | ConvertFrom-Json
        $triggerAuditGeneratedAtForReadinessOrder = [datetime]::MinValue
        $readinessGeneratedAtForTriggerOrder = [datetime]::MinValue

        if ([datetime]::TryParse("$($triggerAuditForReadinessOrder.generatedAt)", [ref]$triggerAuditGeneratedAtForReadinessOrder) -and
            [datetime]::TryParse("$($readinessForTriggerOrder.generatedAt)", [ref]$readinessGeneratedAtForTriggerOrder)) {
            if ($triggerAuditGeneratedAtForReadinessOrder -gt $readinessGeneratedAtForTriggerOrder) {
                $violations.Add('Authority transfer sequence contract violated: trigger audit evidence must be generated before or at cleanup readiness evidence')
            }
        }
    }
    catch {
        $violations.Add("Authority transfer readiness sequence evidence order check failed: reason=$($_.Exception.Message)")
    }
}

if ((Test-Path -LiteralPath $ownershipMigrationArtifactPath) -and (Test-Path -LiteralPath $triggerCleanupCompletionArtifactPath)) {
    try {
        $ownershipForOrder = Get-Content -LiteralPath $ownershipMigrationArtifactPath -Raw -Encoding UTF8 | ConvertFrom-Json
        $completionForOwnershipOrder = Get-Content -LiteralPath $triggerCleanupCompletionArtifactPath -Raw -Encoding UTF8 | ConvertFrom-Json
        $ownershipGeneratedAtOrder = [datetime]::MinValue
        $completionGeneratedAtForOwnershipOrder = [datetime]::MinValue

        if ([datetime]::TryParse("$($ownershipForOrder.generatedAt)", [ref]$ownershipGeneratedAtOrder) -and
            [datetime]::TryParse("$($completionForOwnershipOrder.generatedAt)", [ref]$completionGeneratedAtForOwnershipOrder)) {
            if ($ownershipGeneratedAtOrder -gt $completionGeneratedAtForOwnershipOrder) {
                $violations.Add('Authority migration sequence contract violated: ownership migration evidence must be generated before or at cleanup completion evidence')
            }
        }
    }
    catch {
        $violations.Add("Authority migration sequence evidence order check failed: reason=$($_.Exception.Message)")
    }
}

if ((Test-Path -LiteralPath $triggerAuditArtifactPath) -and (Test-Path -LiteralPath $ownershipMigrationArtifactPath)) {
    try {
        $triggerAuditForOrder = Get-Content -LiteralPath $triggerAuditArtifactPath -Raw -Encoding UTF8 | ConvertFrom-Json
        $ownershipForTriggerOrder = Get-Content -LiteralPath $ownershipMigrationArtifactPath -Raw -Encoding UTF8 | ConvertFrom-Json
        $triggerAuditGeneratedAtOrder = [datetime]::MinValue
        $ownershipGeneratedAtForTriggerOrder = [datetime]::MinValue

        if ([datetime]::TryParse("$($triggerAuditForOrder.generatedAt)", [ref]$triggerAuditGeneratedAtOrder) -and
            [datetime]::TryParse("$($ownershipForTriggerOrder.generatedAt)", [ref]$ownershipGeneratedAtForTriggerOrder)) {
            if ($triggerAuditGeneratedAtOrder -gt $ownershipGeneratedAtForTriggerOrder) {
                $violations.Add('Authority transfer sequence contract violated: trigger audit evidence must be generated before or at ownership migration evidence')
            }
        }
    }
    catch {
        $violations.Add("Authority transfer sequence evidence order check failed: reason=$($_.Exception.Message)")
    }
}

if ((Test-Path -LiteralPath $triggerAuditArtifactPath) -and (Test-Path -LiteralPath $triggerCleanupCompletionArtifactPath)) {
    try {
        $triggerAuditForCompletionOrder = Get-Content -LiteralPath $triggerAuditArtifactPath -Raw -Encoding UTF8 | ConvertFrom-Json
        $completionForTriggerAuditOrder = Get-Content -LiteralPath $triggerCleanupCompletionArtifactPath -Raw -Encoding UTF8 | ConvertFrom-Json
        $triggerAuditGeneratedAtForCompletionOrder = [datetime]::MinValue
        $completionGeneratedAtForTriggerAuditOrder = [datetime]::MinValue

        if ([datetime]::TryParse("$($triggerAuditForCompletionOrder.generatedAt)", [ref]$triggerAuditGeneratedAtForCompletionOrder) -and
            [datetime]::TryParse("$($completionForTriggerAuditOrder.generatedAt)", [ref]$completionGeneratedAtForTriggerAuditOrder)) {
            if ($triggerAuditGeneratedAtForCompletionOrder -gt $completionGeneratedAtForTriggerAuditOrder) {
                $violations.Add('Authority transfer sequence contract violated: trigger audit evidence must be generated before or at cleanup completion evidence')
            }
        }
    }
    catch {
        $violations.Add("Authority transfer completion sequence evidence order check failed: reason=$($_.Exception.Message)")
    }
}

$phaseEvidenceChronology = [ordered]@{
    phase0 = [ordered]@{ artifactPath = $triggerAuditArtifactPath; generatedAt = $null; runId = $null; parseOk = $false }
    phase1 = [ordered]@{ artifactPath = $triggerCleanupReadinessArtifactPath; generatedAt = $null; runId = $null; parseOk = $false }
    phase2 = [ordered]@{ artifactPath = $enforcementAdoptionArtifactPath; generatedAt = $null; runId = $null; parseOk = $false }
    phase3 = [ordered]@{ artifactPath = $facadeBypassArtifactPath; generatedAt = $null; runId = $null; parseOk = $false }
    phase4 = [ordered]@{ artifactPath = $phase4DriftArtifactPath; generatedAt = $null; runId = $null; parseOk = $false }
    phase5 = [ordered]@{ artifactPath = $rollbackCompatibilityArtifactPath; generatedAt = $null; runId = $null; parseOk = $false }
    phase6 = [ordered]@{ artifactPath = $triggerCleanupCompletionArtifactPath; generatedAt = $null; runId = $null; parseOk = $false }
}

foreach ($phaseKey in @('phase0', 'phase1', 'phase2', 'phase3', 'phase4', 'phase5', 'phase6')) {
    $phaseChronologyEntry = $phaseEvidenceChronology[$phaseKey]
    $phaseChronologyPath = "$($phaseChronologyEntry.artifactPath)"

    if (-not (Test-Path -LiteralPath $phaseChronologyPath)) {
        $violations.Add("Phase evidence chronology contract violated: $phaseKey artifact missing: $phaseChronologyPath")
        continue
    }

    try {
        $phaseChronologyReport = Get-Content -LiteralPath $phaseChronologyPath -Raw -Encoding UTF8 | ConvertFrom-Json
        $phaseGeneratedAt = [datetime]::MinValue
        if (-not [datetime]::TryParse("$($phaseChronologyReport.generatedAt)", [ref]$phaseGeneratedAt)) {
            $violations.Add("Phase evidence chronology contract violated: $phaseKey generatedAt parse failed: artifact=$phaseChronologyPath")
            continue
        }

        $phaseChronologyEntry.generatedAt = $phaseGeneratedAt.ToString('o')
        $phaseChronologyEntry.runId = "$($phaseChronologyReport.runId)"
        $phaseChronologyEntry.parseOk = $true
    }
    catch {
        $violations.Add("Phase evidence chronology contract violated: $phaseKey artifact parse failed: artifact=$phaseChronologyPath reason=$($_.Exception.Message)")
    }
}

$phaseChronologyPairs = @(
    @{ earlier = 'phase0'; later = 'phase1' },
    @{ earlier = 'phase0'; later = 'phase2' },
    @{ earlier = 'phase0'; later = 'phase3' },
    @{ earlier = 'phase0'; later = 'phase6' },
    @{ earlier = 'phase1'; later = 'phase6' },
    @{ earlier = 'phase2'; later = 'phase6' },
    @{ earlier = 'phase3'; later = 'phase6' },
    @{ earlier = 'phase4'; later = 'phase6' },
    @{ earlier = 'phase5'; later = 'phase6' }
)

foreach ($phaseChronologyPair in $phaseChronologyPairs) {
    $earlierPhaseEntry = $phaseEvidenceChronology[$phaseChronologyPair.earlier]
    $laterPhaseEntry = $phaseEvidenceChronology[$phaseChronologyPair.later]

    if (-not [bool]$earlierPhaseEntry.parseOk -or -not [bool]$laterPhaseEntry.parseOk) {
        continue
    }

    $earlierRunId = "$($earlierPhaseEntry.runId)"
    $laterRunId = "$($laterPhaseEntry.runId)"
    if ([string]::IsNullOrWhiteSpace($earlierRunId) -or [string]::IsNullOrWhiteSpace($laterRunId) -or
        -not $earlierRunId.Equals($laterRunId, [System.StringComparison]::OrdinalIgnoreCase)) {
        continue
    }

    $earlierPhaseGeneratedAt = [datetime]::Parse("$($earlierPhaseEntry.generatedAt)")
    $laterPhaseGeneratedAt = [datetime]::Parse("$($laterPhaseEntry.generatedAt)")

    if ($earlierPhaseGeneratedAt -gt $laterPhaseGeneratedAt) {
        $violations.Add("Phase evidence chronology contract violated: $($phaseChronologyPair.earlier) generated after $($phaseChronologyPair.later)")
    }
}

if (Test-Path -LiteralPath $backlogResidualArtifactPath) {
    try {
        $backlogResidualFreshness = Get-Content -LiteralPath $backlogResidualArtifactPath -Raw -Encoding UTF8 | ConvertFrom-Json
        $backlogResidualGeneratedAt = [datetime]::MinValue
        if (-not [datetime]::TryParse("$($backlogResidualFreshness.generatedAt)", [ref]$backlogResidualGeneratedAt)) {
            $violations.Add('Backlog residual evidence generatedAt parse failed')
        }
        else {
            $backlogResidualAgeMinutes = ((Get-Date) - $backlogResidualGeneratedAt).TotalMinutes
            if ($backlogResidualAgeMinutes -gt $artifactFreshnessWindowMinutes) {
                $violations.Add("Backlog residual evidence freshness breach: ageMinutes=$([math]::Round($backlogResidualAgeMinutes, 2)) windowMinutes=$artifactFreshnessWindowMinutes")
            }
        }
    }
    catch {
        $violations.Add("Backlog residual evidence freshness parse failed: path=$backlogResidualArtifactPath reason=$($_.Exception.Message)")
    }
}

$report = [ordered]@{
    schema                      = 'bridge_plan_completeness_report_v1'
    generatedAt                 = (Get-Date -Format 'o')
    planPath                    = $planPath
    bridgePolicyPath            = $bridgePolicyPath
    tierRunnerPath              = $tierRunnerPath
    scriptOrder                 = [ordered]@{
        backlogResidualIndex            = $backlogScriptIndex
        triggerSymbolUsageIndex         = $triggerSymbolUsageScriptIndex
        observeShimUsageIndex           = $observeShimUsageScriptIndex
        triggerAstIndex                 = $triggerAstScriptIndex
        triggerAuditIndex               = $triggerAuditScriptIndex
        cleanupPruneIndex               = $cleanupPruneScriptIndex
        cleanupDeferredVerifyIndex      = $cleanupDeferredVerifyScriptIndex
        phase4DriftIndex                = $phase4DriftScriptIndex
        enforcementAdoptionIndex        = $enforcementAdoptionScriptIndex
        enforcementSourcePurityIndex    = $enforcementSourcePurityScriptIndex
        rollbackMatrixIndex             = $rollbackMatrixScriptIndex
        facadeBypassIndex               = $facadeBypassScriptIndex
        canaryNormalizationIndex        = $canaryNormalizationScriptIndex
        metricGovernanceIndex           = $metricGovernanceScriptIndex
        flagDependencyIndex             = $flagDependencyScriptIndex
        cleanupReadinessIndex           = $cleanupReadinessScriptIndex
        ownershipMigrationIndex         = $ownershipMigrationScriptIndex
        cleanupCompletionIndex          = $cleanupCompletionScriptIndex
        bridgeCompletenessIndex         = $bridgeScriptIndex
        backlogBeforeBridgeCompleteness = $scriptOrderSatisfied
    }
    backlogResidualForwarding   = [ordered]@{
        enforceNoSpecFixedForwarded = $backlogEnforceForwardSatisfied
    }
    executionOrderSequence      = [ordered]@{
        sorted = $executionOrderIsSorted
        items  = $executionOrderSequence
    }
    completionDefinitionStatus   = $completionDefinitionStatus
    phaseEvidenceChronology     = $phaseEvidenceChronology
    cleanupReferenceConsistency = [ordered]@{
        backlogPath          = [ordered]@{
            expected  = $expectedBacklogPath
            actual    = $actualBacklogPath
            satisfied = $backlogPathSatisfied
        }
        deferredRegistryPath = [ordered]@{
            expected  = $expectedDeferredRegistryPath
            actual    = $actualDeferredRegistryPath
            satisfied = $deferredRegistryPathSatisfied
        }
        sourceRoot           = [ordered]@{
            expected  = $expectedSourceRoot
            actual    = $actualSourceRoot
            satisfied = $sourceRootSatisfied
        }
    }
    evidenceFreshness           = [ordered]@{
        windowMinutes     = $artifactFreshnessWindowMinutes
        requiredArtifacts = @(
            'trigger_symbol_usage_report.json',
            'observe_shim_usage_report.json',
            'trigger_ast_report.json',
            'validator_tiering_report.json',
            'trigger_cleanup_completion_report.json',
            'backlog_specfixed_residual_report.json',
            'canary_baseline_normalization_report.json',
            'trigger_audit_report.json',
            'metric_governance_report.json',
            'flag_dependency_graph_report.json',
            'enforcement_adoption_report.json',
            'enforcement_source_purity_report.json',
            'trigger_cleanup_readiness_report.json',
            'cleanup_deferred_prune_report.json',
            'cleanup_deferred_report.json',
            'ownership_migration_report.json'
        )
    }
    requiredScriptCount         = $requiredScripts.Count
    phaseCompletionMatrix        = $phaseCompletionMatrix
    artifactStatus              = $artifactStatus
    policyStatus                = $policyStatus
    allowlistStatus             = $allowlistStatus
    violations                  = $violations
}

Set-Content -LiteralPath $reportPath -Value ($report | ConvertTo-Json -Depth 8) -Encoding UTF8
Write-Host "[INFO] bridge plan completeness report written: $reportPath"

if ($violations.Count -gt 0) {
    foreach ($violation in $violations) {
        Write-Host "[ERROR] $violation"
    }
    throw "Bridge runtime migration plan completeness violations detected. count=$($violations.Count)"
}

Write-Host '[PASS] bridge runtime migration plan completeness gate verified'
