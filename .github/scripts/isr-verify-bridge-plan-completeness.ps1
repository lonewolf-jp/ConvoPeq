$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$planPath = Join-Path $repoRoot 'doc\work\bridge_runtime_migration_plan.md'
$tierRunnerPath = Join-Path $repoRoot '.github\scripts\isr-run-tiered-verification.ps1'
$evidenceDir = Join-Path $repoRoot 'evidence'
$reportPath = Join-Path $evidenceDir 'bridge_plan_completeness_report.json'

foreach ($path in @($planPath, $tierRunnerPath)) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Missing required file: $path"
    }
}

if (-not (Test-Path -LiteralPath $evidenceDir)) {
    New-Item -Path $evidenceDir -ItemType Directory | Out-Null
}

$planText = Get-Content -LiteralPath $planPath -Raw -Encoding UTF8
$tierRunnerText = Get-Content -LiteralPath $tierRunnerPath -Raw -Encoding UTF8
$violations = New-Object System.Collections.Generic.List[string]
$artifactFreshnessWindowMinutes = 1440

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

$requiredScripts = @(
    '.github/scripts/isr-trigger-audit.ps1',
    '.github/scripts/isr-verify-validator-tiering.ps1',
    '.github/scripts/isr-verify-policy-top-level-governance.ps1',
    '.github/scripts/isr-verify-workflow-dispatch-input-policy.ps1',
    '.github/scripts/isr-verify-8_1-close-policy.ps1',
    '.github/scripts/isr-verify-8_1-workflow-input-contract.ps1',
    '.github/scripts/isr-verify-8_1-workflow-input-coherence.ps1',
    '.github/scripts/isr-verify-trigger-ast.ps1',
    '.github/scripts/isr-verify-facade-bypass.ps1',
    '.github/scripts/isr-verify-crossfade-observable-state.ps1',
    '.github/scripts/isr-verify-phase4-generation-drift.ps1',
    '.github/scripts/isr-verify-rollback-matrix.ps1',
    '.github/scripts/isr-verify-trigger-cleanup-completion.ps1',
    '.github/scripts/isr-verify-backlog-specfixed-residual.ps1'
)

foreach ($scriptPath in $requiredScripts) {
    if ($tierRunnerText -notmatch [regex]::Escape("'$scriptPath'")) {
        $violations.Add("Tier runner missing plan-completeness script wiring: $scriptPath")
    }
}

$backlogScriptToken = "'.github/scripts/isr-verify-backlog-specfixed-residual.ps1'"
$bridgeScriptToken = "'.github/scripts/isr-verify-bridge-plan-completeness.ps1'"
$backlogScriptIndex = $tierRunnerText.IndexOf($backlogScriptToken)
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

$backlogEnforceForwardAnchor = "elseif (`$scriptPath -eq '.github/scripts/isr-verify-backlog-specfixed-residual.ps1')"
$backlogEnforceForwardCall = '& $scriptPath -EnforceNoSpecFixed'
$backlogEnforceForwardSatisfied = $tierRunnerText.Contains($backlogEnforceForwardAnchor) -and $tierRunnerText.Contains($backlogEnforceForwardCall)
if (-not $backlogEnforceForwardSatisfied) {
    $violations.Add('Tier runner missing EnforceNoSpecFixed forwarding for backlog spec-fixed residual gate')
}

$requiredArtifacts = @(
    @{ Path = (Join-Path $evidenceDir 'trigger_audit_report.json'); Schema = 'trigger_audit_report_v1' },
    @{ Path = (Join-Path $evidenceDir 'validator_tiering_report.json'); Schema = 'validator_tiering_report_v3' },
    @{ Path = (Join-Path $evidenceDir 'policy_top_level_governance_report.json'); Schema = 'policy_top_level_governance_report_v2' },
    @{ Path = (Join-Path $evidenceDir 'workflow_dispatch_input_policy_report.json'); Schema = 'workflow_dispatch_input_policy_report_v1' },
    @{ Path = (Join-Path $evidenceDir 'close_policy_8_1_report.json'); Schema = 'close_policy_8_1_report_v1' },
    @{ Path = (Join-Path $evidenceDir 'close_policy_8_1_workflow_input_contract_report.json'); Schema = 'close_policy_8_1_workflow_input_contract_report_v1' },
    @{ Path = (Join-Path $evidenceDir 'close_policy_8_1_workflow_input_coherence_report.json'); Schema = 'close_policy_8_1_workflow_input_coherence_report_v1' },
    @{ Path = (Join-Path $evidenceDir 'facade_bypass_report.json'); Schema = 'facade_bypass_report_v1' },
    @{ Path = (Join-Path $evidenceDir 'crossfade_observable_state_report.json'); Schema = 'crossfade_observable_state_report_v1' },
    @{ Path = (Join-Path $evidenceDir 'phase4_generation_drift_report.json'); Schema = 'phase4_generation_drift_report_v1' },
    @{ Path = (Join-Path $evidenceDir 'rollback_compatibility_report.json'); Schema = 'rollback_compatibility_report_v1' },
    @{ Path = (Join-Path $evidenceDir 'trigger_cleanup_completion_report.json'); Schema = 'trigger_cleanup_completion_report_v1' },
    @{ Path = (Join-Path $evidenceDir 'backlog_specfixed_residual_report.json'); Schema = 'backlog_specfixed_residual_report_v1' }
)

$requiredPolicies = @(
    @{ Path = (Join-Path $repoRoot '.github\isr-8_1-close-policy.json'); Schema = 'isr_8_1_close_policy_v1' },
    @{ Path = (Join-Path $repoRoot '.github\isr-validator-tiering-policy.json'); Schema = 'isr_validator_tiering_policy_v1' },
    @{ Path = (Join-Path $repoRoot '.github\isr-workflow-dispatch-input-policy.json'); Schema = 'isr_workflow_dispatch_input_policy_v1' }
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

$backlogResidualArtifactPath = Join-Path $evidenceDir 'backlog_specfixed_residual_report.json'
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
    tierRunnerPath              = $tierRunnerPath
    scriptOrder                 = [ordered]@{
        backlogResidualIndex            = $backlogScriptIndex
        bridgeCompletenessIndex         = $bridgeScriptIndex
        backlogBeforeBridgeCompleteness = $scriptOrderSatisfied
    }
    backlogResidualForwarding   = [ordered]@{
        enforceNoSpecFixedForwarded = $backlogEnforceForwardSatisfied
    }
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
            'trigger_cleanup_completion_report.json',
            'backlog_specfixed_residual_report.json'
        )
    }
    requiredScriptCount         = $requiredScripts.Count
    artifactStatus              = $artifactStatus
    policyStatus                = $policyStatus
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
