param(
    [switch]$Apply,
    [string]$TriggerReportPath = 'evidence/trigger_audit_report.json',
    [string]$DeferredRegistryPath = '.github/isr-cleanup-deferred.json',
    [string]$PolicyPath = '.github/isr-cleanup-prune-policy.json'
)

$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$evidenceDir = Join-Path $repoRoot 'evidence'
$reportPath = Join-Path $evidenceDir 'cleanup_deferred_prune_report.json'
$resolvedTriggerReportPath = if ([System.IO.Path]::IsPathRooted($TriggerReportPath)) { $TriggerReportPath } else { Join-Path $repoRoot $TriggerReportPath }
$resolvedDeferredRegistryPath = if ([System.IO.Path]::IsPathRooted($DeferredRegistryPath)) { $DeferredRegistryPath } else { Join-Path $repoRoot $DeferredRegistryPath }
$resolvedPolicyPath = if ([System.IO.Path]::IsPathRooted($PolicyPath)) { $PolicyPath } else { Join-Path $repoRoot $PolicyPath }

foreach ($path in @($resolvedTriggerReportPath, $resolvedDeferredRegistryPath, $resolvedPolicyPath)) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Missing required file: $path"
    }
}

if (-not (Test-Path -LiteralPath $evidenceDir)) {
    New-Item -Path $evidenceDir -ItemType Directory | Out-Null
}

$triggerReport = Get-Content -LiteralPath $resolvedTriggerReportPath -Raw -Encoding UTF8 | ConvertFrom-Json
$deferredRegistry = Get-Content -LiteralPath $resolvedDeferredRegistryPath -Raw -Encoding UTF8 | ConvertFrom-Json
$policy = Get-Content -LiteralPath $resolvedPolicyPath -Raw -Encoding UTF8 | ConvertFrom-Json

if ($triggerReport.schema -ne 'trigger_audit_report_v1') {
    throw "Unexpected trigger audit report schema: $($triggerReport.schema)"
}
if ($deferredRegistry.schema -ne 'cleanup_deferred_registry_v1') {
    throw "Unexpected deferred registry schema: $($deferredRegistry.schema)"
}
if ($policy.schema -ne 'cleanup_prune_policy_v1') {
    throw "Unexpected cleanup prune policy schema: $($policy.schema)"
}

foreach ($field in @('owner', 'issue', 'rationale', 'expiry')) {
    if ($null -eq $policy.$field -or [string]::IsNullOrWhiteSpace("$($policy.$field)")) {
        throw "Cleanup prune policy missing required field: $field"
    }
}

$policyExpiry = [datetime]::ParseExact("$($policy.expiry)", 'yyyy-MM-dd', [System.Globalization.CultureInfo]::InvariantCulture)
if ((Get-Date).Date -gt $policyExpiry.Date) {
    throw "Cleanup prune policy expired: expiry=$($policy.expiry) owner=$($policy.owner) issue=$($policy.issue)"
}

if ($null -eq $policy.readyRule) {
    throw 'Cleanup prune policy missing readyRule'
}

foreach ($field in @('comparator', 'requirePolicyEvaluationNotExpired', 'requireFiniteValues', 'issue')) {
    if ($null -eq $policy.readyRule.$field -or [string]::IsNullOrWhiteSpace("$($policy.readyRule.$field)")) {
        throw "Cleanup prune policy readyRule missing required field: $field"
    }
}

$readyComparator = "$($policy.readyRule.comparator)"
if ($readyComparator -ne 'actual_lte_target_max') {
    throw "Unsupported cleanup prune readyRule comparator: $readyComparator"
}

$requirePolicyEvaluationNotExpired = [bool]$policy.readyRule.requirePolicyEvaluationNotExpired
$requireFiniteValues = [bool]$policy.readyRule.requireFiniteValues

foreach ($field in @('owner', 'issue', 'rationale', 'expiry')) {
    if ($null -eq $deferredRegistry.$field -or [string]::IsNullOrWhiteSpace("$($deferredRegistry.$field)")) {
        throw "Deferred registry missing required field: $field"
    }
}

$registryExpiry = [datetime]::ParseExact("$($deferredRegistry.expiry)", 'yyyy-MM-dd', [System.Globalization.CultureInfo]::InvariantCulture)
if ((Get-Date).Date -gt $registryExpiry.Date) {
    throw "Deferred registry expired: expiry=$($deferredRegistry.expiry) owner=$($deferredRegistry.owner) issue=$($deferredRegistry.issue)"
}

if ($null -eq $triggerReport.policyEvaluations) {
    throw 'Trigger audit report missing policyEvaluations'
}

$evalById = @{}
foreach ($eval in $triggerReport.policyEvaluations) {
    $evalId = "$($eval.id)"
    if ([string]::IsNullOrWhiteSpace($evalId)) {
        continue
    }

    $evalById[$evalId] = $eval
}

$violations = New-Object System.Collections.Generic.List[string]
$readyEntries = New-Object System.Collections.Generic.List[object]
$blockedEntries = New-Object System.Collections.Generic.List[object]
$remainingEntries = New-Object System.Collections.Generic.List[object]

$entries = @($deferredRegistry.entries)
foreach ($entry in $entries) {
    foreach ($field in @('id', 'trigger', 'owner', 'issue', 'rationale', 'expiry')) {
        if ($null -eq $entry.$field -or [string]::IsNullOrWhiteSpace("$($entry.$field)")) {
            $violations.Add("Deferred cleanup entry missing required field: id=$($entry.id) field=$field")
            continue
        }
    }

    $triggerId = "$($entry.trigger)"
    if (-not $evalById.ContainsKey($triggerId)) {
        $violations.Add("Deferred cleanup entry references missing policy evaluation: id=$($entry.id) trigger=$triggerId")
        $remainingEntries.Add($entry) | Out-Null
        continue
    }

    $eval = $evalById[$triggerId]
    if ($null -eq $eval.actual -or $null -eq $eval.targetMax) {
        $violations.Add("Policy evaluation missing actual/targetMax: id=$triggerId")
        $remainingEntries.Add($entry) | Out-Null
        continue
    }

    $evalExpired = [bool]$eval.expired
    if ($requirePolicyEvaluationNotExpired -and $evalExpired) {
        $blockedEntries.Add([ordered]@{
                id = "$($entry.id)"
                trigger = $triggerId
                metric = "$($eval.metric)"
                actual = "$($eval.actual)"
                targetMax = "$($eval.targetMax)"
                owner = "$($entry.owner)"
                issue = "$($entry.issue)"
                reason = 'policy_evaluation_expired'
            }) | Out-Null
        $remainingEntries.Add($entry) | Out-Null
        continue
    }

    [double]$actual = [double]$eval.actual
    [double]$targetMax = [double]$eval.targetMax

    if ($requireFiniteValues -and ((-not [double]::IsFinite($actual)) -or (-not [double]::IsFinite($targetMax)))) {
        $violations.Add("Non-finite cleanup prune values: trigger=$triggerId actual=$actual targetMax=$targetMax")
        $remainingEntries.Add($entry) | Out-Null
        continue
    }

    $candidate = [ordered]@{
        id = "$($entry.id)"
        trigger = $triggerId
        metric = "$($eval.metric)"
        actual = $actual
        targetMax = $targetMax
        owner = "$($entry.owner)"
        issue = "$($entry.issue)"
        reason = 'actual_lte_target_max'
    }

    $isReady = $false
    switch ($readyComparator) {
        'actual_lte_target_max' {
            $isReady = ($actual -le $targetMax)
        }
        default {
            throw "Unsupported cleanup prune comparator at evaluation: $readyComparator"
        }
    }

    if ($isReady) {
        $readyEntries.Add($candidate) | Out-Null
        if (-not $Apply) {
            $remainingEntries.Add($entry) | Out-Null
        }
    }
    else {
        $blockedEntries.Add($candidate) | Out-Null
        $remainingEntries.Add($entry) | Out-Null
    }
}

$prunedCount = 0
if ($Apply -and $readyEntries.Count -gt 0) {
    $newRegistry = [ordered]@{
        schema = "$($deferredRegistry.schema)"
        owner = "$($deferredRegistry.owner)"
        issue = "$($deferredRegistry.issue)"
        rationale = "$($deferredRegistry.rationale)"
        expiry = "$($deferredRegistry.expiry)"
        entries = @($remainingEntries)
    }

    Set-Content -LiteralPath $resolvedDeferredRegistryPath -Value ($newRegistry | ConvertTo-Json -Depth 8) -Encoding UTF8
    $prunedCount = $readyEntries.Count
}

$result = [ordered]@{
    schema = 'cleanup_deferred_prune_report_v1'
    generatedAt = (Get-Date -Format 'o')
    apply = [bool]$Apply
    triggerReportPath = $resolvedTriggerReportPath
    deferredRegistryPath = $resolvedDeferredRegistryPath
    policyPath = $resolvedPolicyPath
    policy = [ordered]@{
        owner = "$($policy.owner)"
        issue = "$($policy.issue)"
        expiry = "$($policy.expiry)"
        readyRule = [ordered]@{
            comparator = $readyComparator
            requirePolicyEvaluationNotExpired = $requirePolicyEvaluationNotExpired
            requireFiniteValues = $requireFiniteValues
            issue = "$($policy.readyRule.issue)"
        }
    }
    readyCount = $readyEntries.Count
    blockedCount = $blockedEntries.Count
    prunedCount = $prunedCount
    remainingCount = $remainingEntries.Count
    readyEntries = $readyEntries
    blockedEntries = $blockedEntries
    violations = $violations
}

Set-Content -LiteralPath $reportPath -Value ($result | ConvertTo-Json -Depth 8) -Encoding UTF8
Write-Host "[INFO] cleanup deferred prune report written: $reportPath"
Write-Host "[INFO] apply=$([bool]$Apply) readyCount=$($readyEntries.Count) blockedCount=$($blockedEntries.Count) prunedCount=$prunedCount remainingCount=$($remainingEntries.Count)"

if ($violations.Count -gt 0) {
    foreach ($violation in $violations) {
        Write-Host "[ERROR] $violation"
    }
    throw "Cleanup deferred prune violations detected. count=$($violations.Count)"
}

Write-Host '[PASS] cleanup deferred prune evaluation completed'
