$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$policyPath = Join-Path $repoRoot ".github\isr-trigger-policy.json"
$symbolAllowlistPath = Join-Path $repoRoot ".github\isr-trigger-symbol-allowlist.json"
$observeShimAllowlistPath = Join-Path $repoRoot ".github\isr-observe-shim-allowlist.json"
$rollbackMatrixPath = Join-Path $repoRoot ".github\isr-rollback-compatibility-matrix.json"
$metricGovernancePath = Join-Path $repoRoot ".github\isr-metric-governance.json"
$tierRunnerPath = Join-Path $repoRoot ".github\scripts\isr-run-tiered-verification.ps1"
$workflowPath = Join-Path $repoRoot ".github\workflows\isr-verification.yml"

if (-not (Test-Path $policyPath)) {
    throw "Missing trigger policy: $policyPath"
}
if (-not (Test-Path $tierRunnerPath)) {
    throw "Missing tiered runner: $tierRunnerPath"
}
if (-not (Test-Path $workflowPath)) {
    throw "Missing workflow: $workflowPath"
}
if (-not (Test-Path $symbolAllowlistPath)) {
    throw "Missing trigger symbol allowlist: $symbolAllowlistPath"
}
if (-not (Test-Path $observeShimAllowlistPath)) {
    throw "Missing observe shim allowlist: $observeShimAllowlistPath"
}
if (-not (Test-Path $rollbackMatrixPath)) {
    throw "Missing rollback compatibility matrix: $rollbackMatrixPath"
}
if (-not (Test-Path $metricGovernancePath)) {
    throw "Missing metric governance registry: $metricGovernancePath"
}

$policy = Get-Content -LiteralPath $policyPath -Raw -Encoding UTF8 | ConvertFrom-Json
if ($policy.schema -ne 'trigger_policy_v1') {
    throw "Unexpected trigger policy schema: $($policy.schema)"
}

foreach ($field in @('owner', 'issue', 'rationale', 'expiry')) {
    if ($null -eq $policy.$field -or [string]::IsNullOrWhiteSpace("$($policy.$field)")) {
        throw "Trigger policy missing required field: $field"
    }
}

$policyExpiry = [datetime]::ParseExact("$($policy.expiry)", 'yyyy-MM-dd', [System.Globalization.CultureInfo]::InvariantCulture)
if ((Get-Date).Date -gt $policyExpiry.Date) {
    throw "Trigger policy expired: expiry=$($policy.expiry) owner=$($policy.owner) issue=$($policy.issue)"
}

if (@('monitor', 'enforce') -notcontains $policy.mode) {
    throw "Unsupported trigger policy mode: $($policy.mode)"
}
if (-not $policy.entries -or $policy.entries.Count -eq 0) {
    throw "Trigger policy entries must be non-empty"
}

$requiredIds = @('activeDspDeletionStart', 'fadingOutDspDeletionStart', 'retireFacadeRemovalStart', 'observeShimRemovalStart', 'runtimeExecutionViewConvergence')
foreach ($requiredId in $requiredIds) {
    $entry = $policy.entries | Where-Object { $_.id -eq $requiredId } | Select-Object -First 1
    if ($null -eq $entry) {
        throw "Missing required trigger policy entry: $requiredId"
    }

    foreach ($field in @('metric', 'targetMax', 'allowedMax', 'owner', 'issue', 'rationale', 'expiry')) {
        if ($null -eq $entry.$field -or [string]::IsNullOrWhiteSpace("$($entry.$field)")) {
            throw "Trigger policy entry '$requiredId' missing required field: $field"
        }
    }
    [double]$targetMax = [double]$entry.targetMax
    [double]$allowedMax = [double]$entry.allowedMax
    if ([double]::IsNaN($targetMax) -or [double]::IsNaN($allowedMax)) {
        throw "Trigger policy entry '$requiredId' must use numeric targetMax/allowedMax values"
    }
    if ($targetMax -gt $allowedMax) {
        throw "Trigger policy entry '$requiredId' has targetMax greater than allowedMax"
    }
}

$tierRunner = Get-Content -LiteralPath $tierRunnerPath -Raw -Encoding UTF8
if ($tierRunner -notmatch 'isr-trigger-audit\.ps1') {
    throw 'Tiered runner does not invoke isr-trigger-audit.ps1'
}
if ($tierRunner -notmatch 'EnforceTriggerPolicy') {
    throw 'Tiered runner missing EnforceTriggerPolicy parameter wiring'
}
if ($tierRunner -notmatch 'isr-verify-trigger-ast\.ps1') {
    throw 'Tiered runner does not invoke isr-verify-trigger-ast.ps1'
}
if ($tierRunner -notmatch 'isr-verify-trigger-symbol-usage\.ps1') {
    throw 'Tiered runner does not invoke isr-verify-trigger-symbol-usage.ps1'
}
if ($tierRunner -notmatch 'isr-verify-observe-shim-usage\.ps1') {
    throw 'Tiered runner does not invoke isr-verify-observe-shim-usage.ps1'
}
if ($tierRunner -notmatch 'isr-verify-rollback-matrix\.ps1') {
    throw 'Tiered runner does not invoke isr-verify-rollback-matrix.ps1'
}
if ($tierRunner -notmatch 'isr-verify-metric-governance\.ps1') {
    throw 'Tiered runner does not invoke isr-verify-metric-governance.ps1'
}
if ($tierRunner -notmatch 'isr-verify-trigger-cleanup-readiness\.ps1') {
    throw 'Tiered runner does not invoke isr-verify-trigger-cleanup-readiness.ps1'
}
if ($tierRunner -notmatch 'isr-verify-enforcement-source-purity\.ps1') {
    throw 'Tiered runner does not invoke isr-verify-enforcement-source-purity.ps1'
}
if ($tierRunner -notmatch 'RequireAstTriggerCheck') {
    throw 'Tiered runner missing RequireAstTriggerCheck parameter wiring'
}
if ($tierRunner -notmatch 'RequireAstEvidence') {
    throw 'Tiered runner missing RequireAstEvidence forwarding wiring'
}

$workflow = Get-Content -LiteralPath $workflowPath -Raw -Encoding UTF8
if ($workflow -notmatch 'enforceTriggerPolicy') {
    throw 'Workflow missing enforceTriggerPolicy input'
}
if ($workflow -notmatch 'EnforceTriggerPolicy') {
    throw 'Workflow missing EnforceTriggerPolicy call path'
}
if ($workflow -notmatch 'requireAstTriggerCheck') {
    throw 'Workflow missing requireAstTriggerCheck input'
}
if ($workflow -notmatch 'RequireAstTriggerCheck') {
    throw 'Workflow missing RequireAstTriggerCheck call path'
}

Write-Host '[PASS] trigger policy governance self-test verified'
