$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$evidenceDir = Join-Path $repoRoot 'evidence'
$reportPath = Join-Path $evidenceDir 'ownership_migration_report.json'

$triggerAuditReportPath = Join-Path $repoRoot 'evidence\trigger_audit_report.json'

if (-not (Test-Path $evidenceDir)) {
    New-Item -Path $evidenceDir -ItemType Directory | Out-Null
}

$headerPath = Join-Path $repoRoot 'src\audioengine\AudioEngine.h'
$lifecyclePath = Join-Path $repoRoot 'src\audioengine\AudioEngine.Processing.DSPCoreLifecycle.cpp'

foreach ($path in @($headerPath, $lifecyclePath, $triggerAuditReportPath)) {
    if (-not (Test-Path $path)) {
        throw "Missing required source file: $path"
    }
}

$headerText = Get-Content -LiteralPath $headerPath -Raw -Encoding UTF8
$lifecycleText = Get-Content -LiteralPath $lifecyclePath -Raw -Encoding UTF8
$triggerAuditReport = Get-Content -LiteralPath $triggerAuditReportPath -Raw -Encoding UTF8 | ConvertFrom-Json

$violations = New-Object System.Collections.Generic.List[string]
$stepDiagnostics = New-Object System.Collections.Generic.List[object]

function Get-SourceLocator {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Text,
        [Parameter(Mandatory = $true)]
        [string]$Pattern,
        [Parameter(Mandatory = $true)]
        [string]$Path,
        [string]$Label = 'unlabeled',
        [switch]$UseRegex
    )

    $index = if ($UseRegex) {
        $match = [regex]::Match($Text, $Pattern)
        if ($match.Success) { $match.Index } else { -1 }
    }
    else {
        $Text.IndexOf($Pattern)
    }

    if ($index -lt 0) {
        return "${Path}:unmatched:${Label}"
    }

    $line = ($Text.Substring(0, $index) -split "`n").Count
    return "${Path}:${line}:${Label}"
}

function Add-StepDiagnostic {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Step,
        [Parameter(Mandatory = $true)]
        [bool]$Satisfied,
        [Parameter(Mandatory = $true)]
        [string]$Reason,
        [Parameter(Mandatory = $true)]
        [string[]]$EvidenceLocators
    )

    $stepDiagnostics.Add([ordered]@{
            step = $Step
            satisfied = $Satisfied
            reason = $Reason
            evidenceLocators = $EvidenceLocators
        }) | Out-Null
}

$newAuthorityIntroduced = $false
$readPathCutoverVerified = $false
$writePathCutoverVerified = $false
$metricsConfirmed = $false
$triggerConfirmed = $false
$legacyAuthorityRemoved = $false

$newAuthorityIntroduced =
    ($headerText -match 'void prepare\(AudioEngine\* ownerEngine,\s*double processingRate,\s*int processingBlockSize\) noexcept') -and
    ($headerText -match 'proc\.setRcuProvider\(\*ownerEngine\);')

Add-StepDiagnostic -Step 'newAuthorityIntroduced' -Satisfied $newAuthorityIntroduced -Reason ($(if ($newAuthorityIntroduced) { 'ConvolverRuntimeState::prepare introduces owner-scoped authority for RCU provider.' } else { 'Owner-scoped authority introduction signatures are missing in ConvolverRuntimeState::prepare.' })) -EvidenceLocators @(
    (Get-SourceLocator -Text $headerText -Pattern 'void prepare\(AudioEngine\* ownerEngine,\s*double processingRate,\s*int processingBlockSize\) noexcept' -Path $headerPath -Label 'prepare_owner_signature' -UseRegex),
    (Get-SourceLocator -Text $headerText -Pattern 'proc\.setRcuProvider\(\*ownerEngine\);' -Path $headerPath -Label 'set_rcu_provider_owner' -UseRegex)
)

if (-not $newAuthorityIntroduced) {
    $violations.Add('Authority step violation: new authority introduction is not verifiable in ConvolverRuntimeState::prepare')
}

$readPathCutoverVerified =
    ($headerText -notmatch 'publishRcuEpoch\(\) noexcept \{ return ownerEngine') -and
    ($headerText -notmatch 'enterRcuReader\(int tid\) noexcept \{ if \(ownerEngine\)') -and
    ($headerText -notmatch 'exitRcuReader\(int tid\) noexcept \{ if \(ownerEngine\)')

Add-StepDiagnostic -Step 'readPathCutoverVerified' -Satisfied $readPathCutoverVerified -Reason ($(if ($readPathCutoverVerified) { 'Read path no longer forwards through legacy ownerEngine accessors.' } else { 'Legacy ownerEngine-based read forwarders are still present.' })) -EvidenceLocators @(
    (Get-SourceLocator -Text $headerText -Pattern 'publishRcuEpoch\(\) noexcept \{ return ownerEngine' -Path $headerPath -Label 'legacy_publish_forwarder' -UseRegex),
    (Get-SourceLocator -Text $headerText -Pattern 'enterRcuReader\(int tid\) noexcept \{ if \(ownerEngine\)' -Path $headerPath -Label 'legacy_enter_forwarder' -UseRegex),
    (Get-SourceLocator -Text $headerText -Pattern 'exitRcuReader\(int tid\) noexcept \{ if \(ownerEngine\)' -Path $headerPath -Label 'legacy_exit_forwarder' -UseRegex)
)

if (-not $readPathCutoverVerified) {
    $violations.Add('Authority step violation: read path cutover is incomplete (ownerEngine forwarders still detected)')
}

$writePathCutoverVerified =
    ($lifecycleText -match 'convolverState->prepare\(owner,\s*processingRate,\s*processingBlockSize\);') -and
    ($lifecycleText -notmatch 'this->ownerEngine\s*=\s*owner;')

Add-StepDiagnostic -Step 'writePathCutoverVerified' -Satisfied $writePathCutoverVerified -Reason ($(if ($writePathCutoverVerified) { 'Write path uses owner pass-through and does not persist legacy owner pointer.' } else { 'Write path cutover contract is violated (owner pass-through missing or legacy assignment remains).' })) -EvidenceLocators @(
    (Get-SourceLocator -Text $lifecycleText -Pattern 'convolverState->prepare\(owner,\s*processingRate,\s*processingBlockSize\);' -Path $lifecyclePath -Label 'prepare_pass_owner' -UseRegex),
    (Get-SourceLocator -Text $lifecycleText -Pattern 'this->ownerEngine\s*=\s*owner;' -Path $lifecyclePath -Label 'legacy_owner_assignment' -UseRegex)
)

if (-not $writePathCutoverVerified) {
    $violations.Add('Authority step violation: write path cutover is incomplete (owner pass-through/assignment contract violated)')
}

$legacyAuthorityRemoved =
    ($headerText -notmatch 'AudioEngine\* ownerEngine\s*=\s*nullptr;') -and
    ($lifecycleText -notmatch '\bownerEngine\b')

Add-StepDiagnostic -Step 'legacyAuthorityRemoved' -Satisfied $legacyAuthorityRemoved -Reason ($(if ($legacyAuthorityRemoved) { 'Legacy owner authority artifacts are absent from header and lifecycle implementation.' } else { 'Legacy owner authority artifacts are still detectable in header/lifecycle.' })) -EvidenceLocators @(
    (Get-SourceLocator -Text $headerText -Pattern 'AudioEngine\* ownerEngine\s*=\s*nullptr;' -Path $headerPath -Label 'legacy_owner_member' -UseRegex),
    (Get-SourceLocator -Text $lifecycleText -Pattern '\bownerEngine\b' -Path $lifecyclePath -Label 'legacy_owner_symbol' -UseRegex)
)

if (-not $legacyAuthorityRemoved) {
    $violations.Add('Authority step violation: legacy authority artifacts still remain in DSP lifecycle/header')
}

if ("$($triggerAuditReport.schema)" -ne 'trigger_audit_report_v1') {
    $violations.Add("Trigger audit schema mismatch for ownership gate: expected=trigger_audit_report_v1 actual=$($triggerAuditReport.schema)")
}

$requiredTriggerMetricFields = @(
    'activeDspRefCount',
    'fadingOutDspWriteCount',
    'runtimeExecutionViewUsageCount'
)

$metricsConfirmed = $true
if ($null -eq $triggerAuditReport.metrics) {
    $metricsConfirmed = $false
    $violations.Add('Authority step violation: trigger audit metrics block missing')
}
else {
    foreach ($metricField in $requiredTriggerMetricFields) {
        if ($null -eq $triggerAuditReport.metrics.$metricField) {
            $metricsConfirmed = $false
            $violations.Add("Authority step violation: trigger audit metric missing for metrics confirmation: field=$metricField")
            continue
        }

        if ([int]$triggerAuditReport.metrics.$metricField -ne 0) {
            $metricsConfirmed = $false
            $violations.Add("Authority step violation: trigger audit metric must be 0 for metrics confirmation: field=$metricField value=$($triggerAuditReport.metrics.$metricField)")
        }
    }
}

Add-StepDiagnostic -Step 'metricsConfirmed' -Satisfied $metricsConfirmed -Reason ($(if ($metricsConfirmed) { 'Trigger audit metrics required for authority transfer are present and zeroed.' } else { 'Trigger audit metrics required for authority transfer are missing or non-zero.' })) -EvidenceLocators @(
    "${triggerAuditReportPath}:metrics.activeDspRefCount:metric_active_dsp_ref",
    "${triggerAuditReportPath}:metrics.fadingOutDspWriteCount:metric_fading_write",
    "${triggerAuditReportPath}:metrics.runtimeExecutionViewUsageCount:metric_runtime_view"
)

$triggerConfirmed = $true
if ($null -eq $triggerAuditReport.policyViolations) {
    $triggerConfirmed = $false
    $violations.Add('Authority step violation: trigger confirmation failed because trigger audit policyViolations field is missing')
}
elseif (@($triggerAuditReport.policyViolations).Count -ne 0) {
    $triggerConfirmed = $false
    $violations.Add("Authority step violation: trigger confirmation failed because policyViolations count is non-zero: count=$(@($triggerAuditReport.policyViolations).Count)")
}

$requiredTriggerEvaluationIds = @(
    'runtimeExecutionViewConvergence',
    'fadingOutDspDeletionStart'
)

$triggerEvaluationById = @{}
foreach ($evaluation in @($triggerAuditReport.policyEvaluations)) {
    $evaluationId = "$($evaluation.id)"
    if (-not [string]::IsNullOrWhiteSpace($evaluationId)) {
        $triggerEvaluationById[$evaluationId] = $evaluation
    }
}

foreach ($requiredTriggerEvaluationId in $requiredTriggerEvaluationIds) {
    if (-not $triggerEvaluationById.ContainsKey($requiredTriggerEvaluationId)) {
        $triggerConfirmed = $false
        $violations.Add("Authority step violation: trigger confirmation missing policy evaluation id=$requiredTriggerEvaluationId")
        continue
    }

    $evaluation = $triggerEvaluationById[$requiredTriggerEvaluationId]
    if ($null -eq $evaluation.expired) {
        $triggerConfirmed = $false
        $violations.Add("Authority step violation: trigger confirmation missing expired field: id=$requiredTriggerEvaluationId")
        continue
    }

    if ([bool]$evaluation.expired) {
        $triggerConfirmed = $false
        $violations.Add("Authority step violation: trigger confirmation encountered expired evaluation: id=$requiredTriggerEvaluationId")
    }

    if ($null -eq $evaluation.actual -or $null -eq $evaluation.allowedMax) {
        $triggerConfirmed = $false
        $violations.Add("Authority step violation: trigger confirmation missing actual/allowedMax: id=$requiredTriggerEvaluationId")
        continue
    }

    if ([double]$evaluation.actual -gt [double]$evaluation.allowedMax) {
        $triggerConfirmed = $false
        $violations.Add("Authority step violation: trigger confirmation actual exceeds allowedMax: id=$requiredTriggerEvaluationId actual=$($evaluation.actual) allowedMax=$($evaluation.allowedMax)")
    }
}

Add-StepDiagnostic -Step 'triggerConfirmed' -Satisfied $triggerConfirmed -Reason ($(if ($triggerConfirmed) { 'Trigger policy evaluations are present, unexpired, and within allowed maxima.' } else { 'Trigger confirmation failed due to missing/expired/exceeding policy evaluations or policy violations.' })) -EvidenceLocators @(
    "${triggerAuditReportPath}:policyViolations:policy_violations",
    "${triggerAuditReportPath}:policyEvaluations.runtimeExecutionViewConvergence:policy_runtime_view",
    "${triggerAuditReportPath}:policyEvaluations.fadingOutDspDeletionStart:policy_fading_write"
)

if ($headerText -match 'AudioEngine\* ownerEngine\s*=\s*nullptr;') {
    $violations.Add('DSPCore must not retain an ownerEngine member')
}

if ($headerText -match 'publishRcuEpoch\(\) noexcept \{ return ownerEngine' -or
    $headerText -match 'enterRcuReader\(int tid\) noexcept \{ if \(ownerEngine\)' -or
    $headerText -match 'exitRcuReader\(int tid\) noexcept \{ if \(ownerEngine\)') {
    $violations.Add('DSPCore must not forward RCU reader methods through ownerEngine')
}

if ($lifecycleText -match 'this->ownerEngine\s*=\s*owner;') {
    $violations.Add('DSPCore::prepare must not store the owner engine pointer')
}

if ($lifecycleText -notmatch 'convolverState->prepare\(owner,\s*processingRate,\s*processingBlockSize\);') {
    $violations.Add('DSPCore::prepare must pass owner directly to convolverState->prepare(owner, ...)')
}

$authorityTransferAllStepsSatisfied =
    $newAuthorityIntroduced -and
    $readPathCutoverVerified -and
    $writePathCutoverVerified -and
    $metricsConfirmed -and
    $triggerConfirmed -and
    $legacyAuthorityRemoved

$report = [ordered]@{
    schema = 'ownership_migration_report_v2'
    generatedAt = (Get-Date -Format 'o')
    headerPath = $headerPath
    lifecyclePath = $lifecyclePath
    triggerAuditReportPath = $triggerAuditReportPath
    authorityTransferSequence = [ordered]@{
        newAuthorityIntroduced = $newAuthorityIntroduced
        readPathCutoverVerified = $readPathCutoverVerified
        writePathCutoverVerified = $writePathCutoverVerified
        metricsConfirmed = $metricsConfirmed
        triggerConfirmed = $triggerConfirmed
        legacyAuthorityRemoved = $legacyAuthorityRemoved
    }
    allStepsSatisfied = $authorityTransferAllStepsSatisfied
    stepDiagnostics = $stepDiagnostics
    violations = $violations
}

Set-Content -LiteralPath $reportPath -Value ($report | ConvertTo-Json -Depth 6) -Encoding UTF8
Write-Host "[INFO] ownership migration report written: $reportPath"

if ($violations.Count -gt 0) {
    foreach ($violation in $violations) {
        Write-Host "[ERROR] $violation"
    }
    throw "Ownership migration violations detected. count=$($violations.Count)"
}

Write-Host '[PASS] ownership migration gate verified'
