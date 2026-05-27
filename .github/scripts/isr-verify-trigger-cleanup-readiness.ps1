param(
    [switch]$EnforceCleanupOnReady,
    [string]$TriggerReportPath = 'evidence/trigger_audit_report.json',
    [string]$DeferredRegistryPath = '.github/isr-cleanup-deferred.json',
    [string]$TriggerPolicyPath = '.github/isr-trigger-policy.json'
)

$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$evidenceDir = Join-Path $repoRoot 'evidence'
$reportPath = Join-Path $evidenceDir 'trigger_cleanup_readiness_report.json'
$resolvedTriggerReportPath = if ([System.IO.Path]::IsPathRooted($TriggerReportPath)) { $TriggerReportPath } else { Join-Path $repoRoot $TriggerReportPath }
$resolvedDeferredRegistryPath = if ([System.IO.Path]::IsPathRooted($DeferredRegistryPath)) { $DeferredRegistryPath } else { Join-Path $repoRoot $DeferredRegistryPath }
$resolvedTriggerPolicyPath = if ([System.IO.Path]::IsPathRooted($TriggerPolicyPath)) { $TriggerPolicyPath } else { Join-Path $repoRoot $TriggerPolicyPath }

foreach ($path in @($resolvedTriggerReportPath, $resolvedDeferredRegistryPath, $resolvedTriggerPolicyPath)) {
    if (-not (Test-Path $path)) {
        throw "Missing required file: $path"
    }
}
if (-not (Test-Path $evidenceDir)) {
    New-Item -Path $evidenceDir -ItemType Directory | Out-Null
}

$triggerReport = Get-Content -LiteralPath $resolvedTriggerReportPath -Raw -Encoding UTF8 | ConvertFrom-Json
$deferredRegistry = Get-Content -LiteralPath $resolvedDeferredRegistryPath -Raw -Encoding UTF8 | ConvertFrom-Json
$triggerPolicy = Get-Content -LiteralPath $resolvedTriggerPolicyPath -Raw -Encoding UTF8 | ConvertFrom-Json

if ($triggerReport.schema -ne 'trigger_audit_report_v1') {
    throw "Unexpected trigger report schema: $($triggerReport.schema)"
}
if ($deferredRegistry.schema -ne 'cleanup_deferred_registry_v1') {
    throw "Unexpected deferred registry schema: $($deferredRegistry.schema)"
}
if ($triggerPolicy.schema -ne 'trigger_policy_v1') {
    throw "Unexpected trigger policy schema: $($triggerPolicy.schema)"
}

foreach ($field in @('owner', 'issue', 'rationale', 'expiry')) {
    if ($null -eq $deferredRegistry.$field -or [string]::IsNullOrWhiteSpace("$($deferredRegistry.$field)")) {
        throw "Deferred registry missing required field: $field"
    }
}

$deferredRegistryExpiry = [datetime]::ParseExact("$($deferredRegistry.expiry)", 'yyyy-MM-dd', [System.Globalization.CultureInfo]::InvariantCulture)
if ((Get-Date).Date -gt $deferredRegistryExpiry.Date) {
    throw "Deferred registry expired: expiry=$($deferredRegistry.expiry) owner=$($deferredRegistry.owner) issue=$($deferredRegistry.issue)"
}

$violations = New-Object System.Collections.Generic.List[string]

if (-not $triggerReport.policyEvaluations) {
    throw 'Trigger audit report missing policyEvaluations'
}

$policyEvalById = @{}
foreach ($eval in $triggerReport.policyEvaluations) {
    $evalId = "$($eval.id)"
    if ([string]::IsNullOrWhiteSpace($evalId)) {
        $violations.Add('Trigger audit policy evaluation contains empty id')
        continue
    }

    $policyEvalById[$evalId] = $eval
}

foreach ($policyEntry in $triggerPolicy.entries) {
    $policyId = "$($policyEntry.id)"
    if (-not $policyEvalById.ContainsKey($policyId)) {
        $violations.Add("Trigger audit missing policy evaluation for trigger id=$policyId")
    }
}

if ($triggerReport.PSObject.Properties.Name -contains 'policyViolations') {
    if ($triggerReport.policyViolations -and $triggerReport.policyViolations.Count -gt 0) {
        foreach ($policyViolation in $triggerReport.policyViolations) {
            $violations.Add("Trigger audit policy violation propagated to cleanup gate: $policyViolation")
        }
    }
}

$targetById = @{}
$metricById = @{}
foreach ($entry in $triggerPolicy.entries) {
    $id = "$($entry.id)"
    $targetById[$id] = [double]$entry.targetMax
    $metricById[$id] = "$($entry.metric)"
}

$ready = New-Object System.Collections.Generic.List[object]
$blocked = New-Object System.Collections.Generic.List[object]

foreach ($entry in $deferredRegistry.entries) {
    foreach ($field in @('id', 'trigger', 'owner', 'issue', 'rationale', 'expiry')) {
        if ($null -eq $entry.$field -or [string]::IsNullOrWhiteSpace("$($entry.$field)")) {
            throw "Deferred cleanup entry '$($entry.id)' missing required field: $field"
        }
    }

    $triggerId = "$($entry.trigger)"
    if (-not $targetById.ContainsKey($triggerId)) {
        $violations.Add("Deferred cleanup entry references unknown trigger: id=$($entry.id) trigger=$triggerId")
        continue
    }

    $metricName = $metricById[$triggerId]
    if ($null -eq $triggerReport.metrics.$metricName) {
        $violations.Add("Trigger report missing metric for deferred cleanup entry: id=$($entry.id) metric=$metricName")
        continue
    }

    [double]$actual = [double]$triggerReport.metrics.$metricName
    [double]$targetMax = [double]$targetById[$triggerId]

    $entryResult = [ordered]@{
        id = "$($entry.id)"
        trigger = $triggerId
        metric = $metricName
        actual = $actual
        targetMax = $targetMax
        owner = "$($entry.owner)"
        issue = "$($entry.issue)"
    }

    if ($actual -le $targetMax) {
        $ready.Add($entryResult) | Out-Null
    }
    else {
        $blocked.Add($entryResult) | Out-Null
    }
}

$result = [ordered]@{
    schema = 'trigger_cleanup_readiness_report_v1'
    generatedAt = (Get-Date -Format 'o')
    enforceCleanupOnReady = [bool]$EnforceCleanupOnReady
    triggerReportPath = $resolvedTriggerReportPath
    deferredRegistryPath = $resolvedDeferredRegistryPath
    triggerPolicyPath = $resolvedTriggerPolicyPath
    deferredRegistryOwner = "$($deferredRegistry.owner)"
    deferredRegistryIssue = "$($deferredRegistry.issue)"
    deferredRegistryExpiry = "$($deferredRegistry.expiry)"
    readyCount = $ready.Count
    blockedCount = $blocked.Count
    ready = $ready
    blocked = $blocked
    violations = $violations
}

$resultJson = $result | ConvertTo-Json -Depth 8
Set-Content -LiteralPath $reportPath -Value $resultJson -Encoding UTF8
Write-Host "[INFO] trigger cleanup readiness report written: $reportPath"
Write-Host "[INFO] readyCount=$($ready.Count) blockedCount=$($blocked.Count)"

if ($violations.Count -gt 0) {
    foreach ($violation in $violations) {
        Write-Host "[ERROR] $violation"
    }
    throw "Trigger cleanup readiness violations detected. count=$($violations.Count)"
}

if ($EnforceCleanupOnReady -and $ready.Count -gt 0) {
    $first = $ready[0]
    throw "Deferred cleanup is ready but not removed: id=$($first.id) trigger=$($first.trigger) metric=$($first.metric) actual=$($first.actual)"
}

Write-Host '[PASS] trigger cleanup readiness gate verified'
