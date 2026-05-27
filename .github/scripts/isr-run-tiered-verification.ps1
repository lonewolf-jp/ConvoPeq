param(
    [ValidateSet('smoke', 'standard', 'exhaustive')]
    [string]$Tier = 'smoke',
    [switch]$RequireRuntimeEvidence,
    [switch]$EnforceTriggerPolicy,
    [switch]$RequireAstTriggerCheck,
    [switch]$RequireClangTidyAudit,
    [switch]$AutoCapture81Log,
    [switch]$Collect81CloseEvidence,
    [int]$Collect81WindowSec = 0,
    [int]$Collect81AutoCaptureTimeoutSec = 0,
    [switch]$Collect81SignalProbe,
    [int]$Collect81ProbeExitMs = 0,
    [switch]$AutoPruneCleanupDeferred,
    [switch]$Enforce81CloseDecision,
    [int]$Enforce81CloseDecisionRetryMax = 0
)

$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
Set-Location $repoRoot

$closePolicyPath = Join-Path $repoRoot '.github\isr-8_1-close-policy.json'
if (-not (Test-Path -LiteralPath $closePolicyPath)) {
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

try {
    $closePolicyExpiry = [datetime]::ParseExact("$($closePolicy.expiry)", 'yyyy-MM-dd', [System.Globalization.CultureInfo]::InvariantCulture)
}
catch {
    throw "8.1 close policy has invalid expiry format: '$($closePolicy.expiry)' (expected yyyy-MM-dd)"
}

if ((Get-Date).Date -gt $closePolicyExpiry.Date) {
    throw "8.1 close policy expired: expiry=$($closePolicy.expiry)"
}

if ($null -eq $closePolicy.collector) {
    throw '8.1 close policy missing collector section'
}

$collectorPolicy = $closePolicy.collector
foreach ($field in @('minWindowSec', 'maxWindowSec', 'minAutoCaptureTimeoutSec', 'maxAutoCaptureTimeoutSec', 'minProbeExitMs', 'maxProbeExitMs', 'minRetryMax', 'maxRetryMax', 'allowedCollectTiers', 'allowedEnforceTiers')) {
    if ($null -eq $collectorPolicy.$field) {
        throw "8.1 close policy collector missing required field: $field"
    }
}

if ($null -eq $closePolicy.workflowInputContract) {
    throw '8.1 close policy missing workflowInputContract section'
}

if ($null -eq $closePolicy.workflowInputContract.inputs) {
    throw '8.1 close policy workflowInputContract missing required field: inputs'
}

function Get-WorkflowInputContractEntry {
    param(
        [pscustomobject]$Policy,
        [string]$InputName
    )

    $entries = @($Policy.workflowInputContract.inputs)
    if ($entries.Count -eq 0) {
        throw '8.1 close policy workflowInputContract requires non-empty inputs'
    }

    $entry = $entries | Where-Object { "$($_.name)" -eq $InputName } | Select-Object -First 1
    if ($null -eq $entry) {
        throw "8.1 close policy workflowInputContract missing input contract entry: $InputName"
    }

    return $entry
}

function Resolve-WorkflowInputContractIntDefault {
    param(
        [pscustomobject]$Policy,
        [string]$InputName
    )

    $entry = Get-WorkflowInputContractEntry -Policy $Policy -InputName $InputName
    if ("$($entry.type)" -ne 'string') {
        throw "8.1 close policy workflowInputContract input must be type=string for int default resolution: $InputName"
    }

    $rawDefault = "$($entry.default)"
    $parsed = 0
    if (-not [int]::TryParse($rawDefault, [ref]$parsed) -or $parsed -le 0) {
        throw "8.1 close policy workflowInputContract input has invalid positive integer default: name=$InputName default='$rawDefault'"
    }

    return $parsed
}

if (-not $PSBoundParameters.ContainsKey('Collect81WindowSec')) {
    $Collect81WindowSec = Resolve-WorkflowInputContractIntDefault -Policy $closePolicy -InputName 'collect81WindowSec'
}

if (-not $PSBoundParameters.ContainsKey('Collect81AutoCaptureTimeoutSec')) {
    $Collect81AutoCaptureTimeoutSec = Resolve-WorkflowInputContractIntDefault -Policy $closePolicy -InputName 'collect81AutoCaptureTimeoutSec'
}

if (-not $PSBoundParameters.ContainsKey('Collect81ProbeExitMs')) {
    $Collect81ProbeExitMs = Resolve-WorkflowInputContractIntDefault -Policy $closePolicy -InputName 'collect81ProbeExitMs'
}

if (-not $PSBoundParameters.ContainsKey('Enforce81CloseDecisionRetryMax')) {
    $Enforce81CloseDecisionRetryMax = Resolve-WorkflowInputContractIntDefault -Policy $closePolicy -InputName 'enforce81CloseDecisionRetryMax'
}

if ($Enforce81CloseDecision -and -not $Collect81CloseEvidence) {
    throw 'Invalid configuration: Enforce81CloseDecision requires Collect81CloseEvidence.'
}

if ($Enforce81CloseDecisionRetryMax -lt 1) {
    throw 'Invalid configuration: Enforce81CloseDecisionRetryMax must be >= 1.'
}

if ($AutoPruneCleanupDeferred -and $Tier -eq 'smoke') {
    throw 'Invalid configuration: AutoPruneCleanupDeferred requires Tier=standard or exhaustive.'
}

$allowedCollectTiers = @($collectorPolicy.allowedCollectTiers)
$allowedEnforceTiers = @($collectorPolicy.allowedEnforceTiers)

if ($Collect81CloseEvidence -and ($allowedCollectTiers -notcontains $Tier)) {
    throw "Invalid configuration: Collect81CloseEvidence is not allowed for Tier=$Tier by policy"
}

if ($Enforce81CloseDecision -and ($allowedEnforceTiers -notcontains $Tier)) {
    throw "Invalid configuration: Enforce81CloseDecision is not allowed for Tier=$Tier by policy"
}

if ($Collect81WindowSec -lt [int]$collectorPolicy.minWindowSec -or $Collect81WindowSec -gt [int]$collectorPolicy.maxWindowSec) {
    throw "Invalid configuration: Collect81WindowSec must be between $($collectorPolicy.minWindowSec) and $($collectorPolicy.maxWindowSec)."
}

if ($Collect81AutoCaptureTimeoutSec -lt [int]$collectorPolicy.minAutoCaptureTimeoutSec -or $Collect81AutoCaptureTimeoutSec -gt [int]$collectorPolicy.maxAutoCaptureTimeoutSec) {
    throw "Invalid configuration: Collect81AutoCaptureTimeoutSec must be between $($collectorPolicy.minAutoCaptureTimeoutSec) and $($collectorPolicy.maxAutoCaptureTimeoutSec)."
}

if ($Collect81ProbeExitMs -lt [int]$collectorPolicy.minProbeExitMs -or $Collect81ProbeExitMs -gt [int]$collectorPolicy.maxProbeExitMs) {
    throw "Invalid configuration: Collect81ProbeExitMs must be between $($collectorPolicy.minProbeExitMs) and $($collectorPolicy.maxProbeExitMs)."
}

if ($Enforce81CloseDecisionRetryMax -lt [int]$collectorPolicy.minRetryMax -or $Enforce81CloseDecisionRetryMax -gt [int]$collectorPolicy.maxRetryMax) {
    throw "Invalid configuration: Enforce81CloseDecisionRetryMax must be between $($collectorPolicy.minRetryMax) and $($collectorPolicy.maxRetryMax)."
}

if ($RequireRuntimeEvidence) {
    $env:ISR_REQUIRE_RUNTIME_EVIDENCE = '1'
    if ([string]::IsNullOrWhiteSpace($env:CONVO_ISR_RUNTIME_RUN_ID)) {
        $env:CONVO_ISR_RUNTIME_RUN_ID = "local-$([guid]::NewGuid().ToString('N'))"
    }
    Write-Host "[INFO] tier=$Tier strict runtime evidence mode enabled: runId=$($env:CONVO_ISR_RUNTIME_RUN_ID)"
    & ./.github/scripts/isr-run-runtime-evidence.ps1
}
else {
    $env:ISR_REQUIRE_RUNTIME_EVIDENCE = '0'
    Remove-Item Env:CONVO_ISR_RUNTIME_RUN_ID -ErrorAction SilentlyContinue
    & ./.github/scripts/isr-seed-evidence.ps1
}

$smokeScripts = @(
    '.github/scripts/isr-verify-v1-immutability.ps1',
    '.github/scripts/isr-verify-v2-seal.ps1',
    '.github/scripts/isr-verify-v3-runtime-graph-immutability.ps1',
    '.github/scripts/isr-verify-v4-dsp-handle-policy.ps1',
    '.github/scripts/isr-verify-v5-retire-authority-lane.ps1',
    '.github/scripts/isr-verify-v6-domain-f-ordering.ps1',
    '.github/scripts/isr-verify-v7-rt-nonrt-retire-bridge.ps1',
    '.github/scripts/isr-verify-v8-shared-split-readiness.ps1',
    '.github/scripts/isr-verify-phase4-generation-drift.ps1',
    '.github/scripts/isr-verify-v6.ps1',
    '.github/scripts/isr-verify-workflow-dispatch-input-policy.ps1',
    '.github/scripts/isr-verify-gate-wiring.ps1'
)

$standardAdditionalScripts = @(
    '.github/scripts/isr-verify-v3.ps1',
    '.github/scripts/isr-verify-v4.ps1',
    '.github/scripts/isr-verify-v5.ps1',
    '.github/scripts/isr-verify-v7.ps1',
    '.github/scripts/isr-verify-v8.ps1',
    '.github/scripts/isr-verify-v9.ps1',
    '.github/scripts/isr-verify-v10.ps1',
    '.github/scripts/isr-verify-v10-ownership-cycle.ps1',
    '.github/scripts/isr-verify-evidence-provenance.ps1',
    '.github/scripts/isr-verify-runtime-reduction-gate.ps1',
    '.github/scripts/isr-verify-proof-scope.ps1',
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
    '.github/scripts/isr-verify-8_1-workflow-input-contract.ps1',
    '.github/scripts/isr-verify-8_1-workflow-input-coherence.ps1',
    '.github/scripts/isr-verify-policy-top-level-governance.ps1',
    '.github/scripts/isr-verify-rtmutable-boundary.ps1',
    '.github/scripts/isr-verify-facade-bypass.ps1',
    '.github/scripts/isr-verify-latency-alignment.ps1',
    '.github/scripts/isr-verify-crossfade-observable-state.ps1',
    '.github/scripts/isr-verify-canary-baseline-normalization.ps1',
    '.github/scripts/isr-verify-ownership-migration.ps1',
    '.github/scripts/isr-verify-validator-tiering.ps1',
    '.github/scripts/isr-verify-trigger-cleanup-completion.ps1',
    '.github/scripts/isr-verify-backlog-specfixed-residual.ps1',
    '.github/scripts/isr-verify-bridge-plan-completeness.ps1',
    '.github/scripts/isr-verify-clang-tidy-readiness.ps1',
    '.github/scripts/isr-verify-clang-tidy-audit.ps1',
    '.github/scripts/check-src-atomic-dotcall.ps1',
    '.github/scripts/check-list-compliance.ps1',
    '.github/scripts/isr-verify-p3-governance.ps1'
)

$exhaustiveAdditionalScripts = @(
    '.github/scripts/isr-verify-v5.ps1',
    '.github/scripts/isr-verify-v6.ps1',
    '.github/scripts/isr-verify-v7.ps1',
    '.github/scripts/isr-verify-v8.ps1',
    '.github/scripts/isr-verify-v9.ps1'
)

$scriptsToRun = @($smokeScripts)
if ($Tier -eq 'standard' -or $Tier -eq 'exhaustive') {
    $scriptsToRun += $standardAdditionalScripts
}
if ($Tier -eq 'exhaustive') {
    $scriptsToRun += $exhaustiveAdditionalScripts
}

foreach ($scriptPath in $scriptsToRun) {
    if (-not (Test-Path $scriptPath)) {
        throw "Missing verification script: $scriptPath"
    }

    Write-Host "[INFO] tier=$Tier run: $scriptPath"
    if ($scriptPath -eq '.github/scripts/isr-trigger-audit.ps1') {
        if ($EnforceTriggerPolicy -and $RequireAstTriggerCheck) {
            & $scriptPath -Enforce -RequireAstEvidence
        }
        elseif ($EnforceTriggerPolicy) {
            & $scriptPath -Enforce
        }
        elseif ($RequireAstTriggerCheck) {
            & $scriptPath -RequireAstEvidence
        }
        else {
            & $scriptPath
        }
    }
    elseif ($scriptPath -eq '.github/scripts/isr-verify-trigger-ast.ps1') {
        if ($RequireAstTriggerCheck) {
            & $scriptPath -RequireAst
        }
        else {
            & $scriptPath
        }
    }
    elseif ($scriptPath -eq '.github/scripts/isr-verify-trigger-cleanup-readiness.ps1') {
        & $scriptPath -EnforceCleanupOnReady
    }
    elseif ($scriptPath -eq '.github/scripts/isr-prune-cleanup-deferred.ps1') {
        if ($AutoPruneCleanupDeferred) {
            & $scriptPath -Apply
        }
        else {
            & $scriptPath
        }
    }
    elseif ($scriptPath -eq '.github/scripts/isr-verify-enforcement-adoption.ps1') {
        & $scriptPath -Tier $Tier
    }
    elseif ($scriptPath -eq '.github/scripts/isr-verify-enforcement-source-purity.ps1') {
        if ($RequireAstTriggerCheck) {
            & $scriptPath -RequireAstEvidence
        }
        else {
            & $scriptPath
        }
    }
    elseif ($scriptPath -eq '.github/scripts/isr-verify-trigger-cleanup-completion.ps1') {
        if ($RequireAstTriggerCheck) {
            & $scriptPath -RequireAstEvidence
        }
        else {
            & $scriptPath
        }
    }
    elseif ($scriptPath -eq '.github/scripts/isr-verify-backlog-specfixed-residual.ps1') {
        & $scriptPath -EnforceNoSpecFixed
    }
    elseif ($scriptPath -eq '.github/scripts/isr-verify-clang-tidy-audit.ps1') {
        if ($RequireClangTidyAudit) {
            & $scriptPath -Tier $Tier -RequireClangTidy
        }
        else {
            & $scriptPath -Tier $Tier
        }
    }
    elseif ($scriptPath -eq '.github/scripts/isr-verify-8_1-close-policy.ps1') {
        & $scriptPath -Tier $Tier
    }
    elseif ($scriptPath -eq '.github/scripts/isr-rebuild-admission-8_1-metrics.ps1') {
        if ($AutoCapture81Log) {
            & $scriptPath -TryAutoCaptureOnMissingLog
        }
        else {
            & $scriptPath
        }
    }
    else {
        & $scriptPath
    }
}

if ($Collect81CloseEvidence -and ($Tier -eq 'standard' -or $Tier -eq 'exhaustive')) {
    $collectorScriptPath = '.github/scripts/isr-collect-rebuild-admission-8_1-close-evidence.ps1'
    $collectorReportPath = 'evidence/rebuild_admission_8_1_close_collection_report.json'
    $max81CloseDecisionAttempts = if ($Enforce81CloseDecision) { $Enforce81CloseDecisionRetryMax } else { 1 }
    $attempt81CloseDecision = 1
    if (-not (Test-Path $collectorScriptPath)) {
        throw "Missing 8.1 close evidence collector script: $collectorScriptPath"
    }

    while ($attempt81CloseDecision -le $max81CloseDecisionAttempts) {
        Write-Host "[INFO] tier=$Tier run: $collectorScriptPath (attempt=$attempt81CloseDecision/$max81CloseDecisionAttempts)"
        if ($Collect81SignalProbe) {
            & $collectorScriptPath -WindowSec $Collect81WindowSec -AutoCaptureTimeoutSec $Collect81AutoCaptureTimeoutSec -ProbeOnInsufficientSignals -ProbeExitMs $Collect81ProbeExitMs
        }
        else {
            & $collectorScriptPath -WindowSec $Collect81WindowSec -AutoCaptureTimeoutSec $Collect81AutoCaptureTimeoutSec
        }

        if (-not $Enforce81CloseDecision) {
            break
        }

        if (-not (Test-Path -LiteralPath $collectorReportPath)) {
            throw "8.1 close decision enforce failed: missing collector report: $collectorReportPath"
        }

        $collectorReport = Get-Content -LiteralPath $collectorReportPath -Raw -Encoding UTF8 | ConvertFrom-Json
        if ($null -eq $collectorReport.operationalDecision) {
            throw '8.1 close decision enforce failed: operationalDecision is missing in collector report'
        }

        $policyVersion = [string]$collectorReport.operationalDecision.decisionPolicyVersion
        if ($policyVersion -ne '8.1-close-ops-v3') {
            throw "8.1 close decision enforce failed: unexpected decisionPolicyVersion=$policyVersion"
        }

        $decisionReady = [bool]$collectorReport.operationalDecision.closeReady
        $decisionSource = [string]$collectorReport.operationalDecision.source
        $allowedSources = @('probeDelta', 'delta', 'baseline')
        if ($allowedSources -notcontains $decisionSource) {
            throw "8.1 close decision enforce failed: unexpected source=$decisionSource"
        }

        $sourceCandidates = @($collectorReport.operationalDecision.sourceCandidates)
        if ($sourceCandidates.Count -lt 3 -or @($sourceCandidates | Select-Object -First 3) -join ',' -ne 'probeDelta,delta,baseline') {
            throw '8.1 close decision enforce failed: sourceCandidates contract mismatch'
        }

        $blockingSignals = @($collectorReport.operationalDecision.blockingSignals)
        if (-not $decisionReady) {
            $blockingText = if ($blockingSignals.Count -gt 0) { ($blockingSignals -join ', ') } else { 'unknown' }
            $retryableBlockingSignal = $blockingSignals.Count -eq 1 -and $blockingSignals[0] -eq 'timeoutForcedDispatchSeen'
            if ($retryableBlockingSignal -and $attempt81CloseDecision -lt $max81CloseDecisionAttempts) {
                Write-Host "[WARN] 8.1 close decision transient signal detected (timeoutForcedDispatchSeen). Retrying collector once."
                $attempt81CloseDecision++
                continue
            }

            throw "8.1 close decision enforce failed: source=$decisionSource blockingSignals=$blockingText"
        }

        Write-Host "[PASS] 8.1 close operational decision enforced: policyVersion=$policyVersion source=$decisionSource closeReady=true"
        break
    }
}

Write-Host "[PASS] tiered verification completed. tier=$Tier"
