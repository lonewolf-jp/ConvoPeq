param(
    [switch]$Enforce,
    [switch]$RequireAstEvidence,
    [string]$PolicyPath = '.github/isr-trigger-policy.json'
)

$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$evidenceDir = Join-Path $repoRoot "evidence"
$reportPath = Join-Path $evidenceDir "trigger_audit_report.json"
$symbolUsageReportPath = Join-Path $evidenceDir "trigger_symbol_usage_report.json"
$observeShimReportPath = Join-Path $evidenceDir "observe_shim_usage_report.json"
$triggerAstReportPath = Join-Path $evidenceDir "trigger_ast_report.json"
$resolvedPolicyPath = if ([System.IO.Path]::IsPathRooted($PolicyPath)) { $PolicyPath } else { Join-Path $repoRoot $PolicyPath }

function Get-SymbolBlockedMatches {
    param(
        [object]$SymbolReport,
        [string]$SymbolName
    )

    if ($null -eq $SymbolReport -or $null -eq $SymbolReport.symbolStats) {
        return $null
    }

    foreach ($entry in $SymbolReport.symbolStats) {
        if ("$($entry.symbol)" -eq $SymbolName -and $null -ne $entry.blockedMatches) {
            return [int]$entry.blockedMatches
        }
    }

    return $null
}

function Get-SymbolTotalMatches {
    param(
        [object]$SymbolReport,
        [string]$SymbolName
    )

    if ($null -eq $SymbolReport -or $null -eq $SymbolReport.symbolStats) {
        return $null
    }

    foreach ($entry in $SymbolReport.symbolStats) {
        if ("$($entry.symbol)" -eq $SymbolName -and $null -ne $entry.totalMatches) {
            return [int]$entry.totalMatches
        }
    }

    return $null
}

function Test-SourcePrefix {
    param(
        [string]$Source,
        [string[]]$AllowedPrefixes
    )

    foreach ($prefix in $AllowedPrefixes) {
        if ($Source -match ("^{0}" -f [regex]::Escape($prefix))) {
            return $true
        }
    }

    return $false
}

if (-not (Test-Path $resolvedPolicyPath)) {
    throw "Missing trigger policy: $resolvedPolicyPath"
}

if (-not (Test-Path $evidenceDir)) {
    New-Item -Path $evidenceDir -ItemType Directory | Out-Null
}

# Trigger candidates from bridge_runtime_migration_plan.md section 9.1
$activeDspRawRefCount = 0
$activeDspRefCount = 0
$activeDspMetricSource = 'symbolBlockedMatchesRefreshed'

$retireFacadeRawDependencyCount = 0
$retireFacadeDirectDependencyCount = 0
$retireFacadeMetricSource = 'symbolBlockedMatchesRefreshed'
$retireFacadeRuntimeExecutionCount = 0
$retireFacadeRuntimeExecutionMetricSource = 'symbolTotalMatchesRefreshed'
$runtimeExecutionViewUsageCount = 0
$runtimeExecutionViewMetricSource = 'symbolBlockedMatchesRefreshed'

$legacyDirectObserveRawCount = 0
$legacyDirectObserveUsageCount = 0
$legacyDirectObserveMetricSource = 'observeShimBlockedMatches'

$fadingOutDspWriteCount = 0
$fadingOutDspMetricSource = 'triggerAstEffectiveMatches'

$symbolUsageScriptPath = Join-Path $repoRoot '.github\scripts\isr-verify-trigger-symbol-usage.ps1'
$observeShimScriptPath = Join-Path $repoRoot '.github\scripts\isr-verify-observe-shim-usage.ps1'
$triggerAstScriptPath = Join-Path $repoRoot '.github\scripts\isr-verify-trigger-ast.ps1'

foreach ($scriptPath in @($symbolUsageScriptPath, $observeShimScriptPath, $triggerAstScriptPath)) {
    if (-not (Test-Path $scriptPath)) {
        throw "Missing required evidence generator script: $scriptPath"
    }

    if ($scriptPath -eq $triggerAstScriptPath -and $RequireAstEvidence) {
        & pwsh -NoProfile -ExecutionPolicy Bypass -File $scriptPath -RequireAst | Out-Null
    }
    else {
        & pwsh -NoProfile -ExecutionPolicy Bypass -File $scriptPath | Out-Null
    }
}

if (-not (Test-Path $symbolUsageReportPath)) {
    throw "Missing required evidence report: $symbolUsageReportPath"
}

$symbolUsageReport = Get-Content -LiteralPath $symbolUsageReportPath -Raw -Encoding UTF8 | ConvertFrom-Json
if ($symbolUsageReport.schema -ne 'trigger_symbol_usage_report_v1') {
    throw "Unexpected trigger symbol usage report schema: $($symbolUsageReport.schema)"
}

if ($null -eq $symbolUsageReport.symbolStats) {
    throw 'Trigger symbol usage report missing symbolStats'
}

$activeDspBlockedRefreshed = Get-SymbolBlockedMatches -SymbolReport $symbolUsageReport -SymbolName 'activeDSP'
if ($null -eq $activeDspBlockedRefreshed) {
    throw 'Trigger symbol usage report missing symbolStats entry: activeDSP'
}
$activeDspRefCount = [int]$activeDspBlockedRefreshed

$activeDspTotalRefreshed = Get-SymbolTotalMatches -SymbolReport $symbolUsageReport -SymbolName 'activeDSP'
if ($null -eq $activeDspTotalRefreshed) {
    throw 'Trigger symbol usage report missing totalMatches entry: activeDSP'
}
$activeDspRawRefCount = [int]$activeDspTotalRefreshed

$retireMemberBlockedRefreshed = Get-SymbolBlockedMatches -SymbolReport $symbolUsageReport -SymbolName 'runtimePublicationCoordinator_'
$retireFactoryBlockedRefreshed = Get-SymbolBlockedMatches -SymbolReport $symbolUsageReport -SymbolName 'RuntimePublicationCoordinator::create'
if ($null -eq $retireMemberBlockedRefreshed -or $null -eq $retireFactoryBlockedRefreshed) {
    throw 'Trigger symbol usage report missing symbolStats entries for retire facade metrics'
}
$retireFacadeDirectDependencyCount = [int]$retireMemberBlockedRefreshed + [int]$retireFactoryBlockedRefreshed

$retireMemberTotalRefreshed = Get-SymbolTotalMatches -SymbolReport $symbolUsageReport -SymbolName 'runtimePublicationCoordinator_'
if ($null -eq $retireMemberTotalRefreshed) {
    throw 'Trigger symbol usage report missing totalMatches entry: runtimePublicationCoordinator_'
}

$retireFactoryTotalRefreshed = Get-SymbolTotalMatches -SymbolReport $symbolUsageReport -SymbolName 'RuntimePublicationCoordinator::create'
if ($null -eq $retireFactoryTotalRefreshed) {
    throw 'Trigger symbol usage report missing totalMatches entry: RuntimePublicationCoordinator::create'
}
$retireFacadeRuntimeExecutionCount = [int]$retireFactoryTotalRefreshed
$retireFacadeRawDependencyCount = [int]$retireMemberTotalRefreshed + [int]$retireFactoryTotalRefreshed

$runtimeExecutionViewBlockedRefreshed = Get-SymbolBlockedMatches -SymbolReport $symbolUsageReport -SymbolName 'RuntimeExecutionView'
if ($null -eq $runtimeExecutionViewBlockedRefreshed) {
    throw 'Trigger symbol usage report missing symbolStats entry: RuntimeExecutionView'
}
$runtimeExecutionViewUsageCount = [int]$runtimeExecutionViewBlockedRefreshed

if (-not (Test-Path $observeShimReportPath)) {
    throw "Missing required evidence report: $observeShimReportPath"
}

$observeShimReport = Get-Content -LiteralPath $observeShimReportPath -Raw -Encoding UTF8 | ConvertFrom-Json
if ($observeShimReport.schema -ne 'observe_shim_usage_report_v1') {
    throw "Unexpected observe shim usage report schema: $($observeShimReport.schema)"
}

if ($null -eq $observeShimReport.blockedMatches) {
    throw 'Observe shim usage report missing blockedMatches'
}

if ($null -eq $observeShimReport.totalMatches) {
    throw 'Observe shim usage report missing totalMatches'
}

$legacyDirectObserveUsageCount = [int]$observeShimReport.blockedMatches
$legacyDirectObserveRawCount = [int]$observeShimReport.totalMatches

if (-not (Test-Path $triggerAstReportPath)) {
    throw "Missing required evidence report: $triggerAstReportPath"
}

$triggerAstReport = Get-Content -LiteralPath $triggerAstReportPath -Raw -Encoding UTF8 | ConvertFrom-Json
if ($triggerAstReport.schema -ne 'trigger_ast_report_v1') {
    throw "Unexpected trigger AST report schema: $($triggerAstReport.schema)"
}

if ($null -eq $triggerAstReport.available -or -not [bool]$triggerAstReport.available) {
    throw 'Trigger AST report must be available=true for trigger audit'
}

if ($null -eq $triggerAstReport.commandOk -or -not [bool]$triggerAstReport.commandOk) {
    throw 'Trigger AST report must be commandOk=true for trigger audit'
}

if ($null -eq $triggerAstReport.fadingOutDspWriteEffectiveMatches) {
    throw 'Trigger AST report missing fadingOutDspWriteEffectiveMatches'
}

if ($null -eq $triggerAstReport.fadingOutDspWriteEffectiveSource) {
    throw 'Trigger AST report missing fadingOutDspWriteEffectiveSource'
}

if ($RequireAstEvidence) {
    if ($null -eq $triggerAstReport.required -or -not [bool]$triggerAstReport.required) {
        throw 'Trigger AST report must be required=true when trigger audit RequireAstEvidence is specified'
    }

    if ("$($triggerAstReport.fadingOutDspWriteEffectiveSource)" -ne 'astOnly') {
        throw "Trigger AST report effective source must be astOnly when RequireAstEvidence is specified. actual=$($triggerAstReport.fadingOutDspWriteEffectiveSource)"
    }
}

$fadingOutDspWriteCount = [int]$triggerAstReport.fadingOutDspWriteEffectiveMatches

$sourceViolations = New-Object System.Collections.Generic.List[string]

if (-not (Test-SourcePrefix -Source $activeDspMetricSource -AllowedPrefixes @('symbol'))) {
    $sourceViolations.Add("activeDspMetricSource must be symbol* source. actual=$activeDspMetricSource")
}

if (-not (Test-SourcePrefix -Source $retireFacadeMetricSource -AllowedPrefixes @('symbol'))) {
    $sourceViolations.Add("retireFacadeMetricSource must be symbol* source. actual=$retireFacadeMetricSource")
}

if (-not (Test-SourcePrefix -Source $retireFacadeRuntimeExecutionMetricSource -AllowedPrefixes @('symbol'))) {
    $sourceViolations.Add("retireFacadeRuntimeExecutionMetricSource must be symbol* source. actual=$retireFacadeRuntimeExecutionMetricSource")
}

if (-not (Test-SourcePrefix -Source $runtimeExecutionViewMetricSource -AllowedPrefixes @('symbol'))) {
    $sourceViolations.Add("runtimeExecutionViewMetricSource must be symbol* source. actual=$runtimeExecutionViewMetricSource")
}

if (-not (Test-SourcePrefix -Source $legacyDirectObserveMetricSource -AllowedPrefixes @('observeShim'))) {
    $sourceViolations.Add("legacyDirectObserveMetricSource must be observeShim* source. actual=$legacyDirectObserveMetricSource")
}

if (-not (Test-SourcePrefix -Source $fadingOutDspMetricSource -AllowedPrefixes @('triggerAst'))) {
    $sourceViolations.Add("fadingOutDspMetricSource must be triggerAst* source. actual=$fadingOutDspMetricSource")
}

if ($sourceViolations.Count -gt 0) {
    foreach ($violation in $sourceViolations) {
        Write-Host "[ERROR] Trigger audit source contract violated: $violation"
    }

    throw "Trigger audit source contract violations detected. count=$($sourceViolations.Count)"
}

$metrics = [ordered]@{
    activeDspRefCount                 = $activeDspRefCount
    activeDspRawRefCount              = $activeDspRawRefCount
    fadingOutDspWriteCount            = $fadingOutDspWriteCount
    retireFacadeDirectDependencyCount = $retireFacadeDirectDependencyCount
    retireFacadeRawDependencyCount    = $retireFacadeRawDependencyCount
    retireFacadeRuntimeExecutionCount = $retireFacadeRuntimeExecutionCount
    runtimeExecutionViewUsageCount    = $runtimeExecutionViewUsageCount
    legacyDirectObserveUsageCount     = $legacyDirectObserveUsageCount
    legacyDirectObserveRawCount       = $legacyDirectObserveRawCount
}

$policy = Get-Content -LiteralPath $resolvedPolicyPath -Raw -Encoding UTF8 | ConvertFrom-Json
if ($policy.schema -ne 'trigger_policy_v1') {
    throw "Unsupported trigger policy schema: $($policy.schema)"
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

$policyMode = "$($policy.mode)"
if (@('monitor', 'enforce') -notcontains $policyMode) {
    throw "Unsupported trigger policy mode: $policyMode"
}

$policyViolations = New-Object System.Collections.Generic.List[string]
$entryEvaluations = New-Object System.Collections.Generic.List[object]
$today = (Get-Date).Date

foreach ($entry in $policy.entries) {
    foreach ($field in @('id', 'metric', 'targetMax', 'allowedMax', 'owner', 'issue', 'rationale', 'expiry')) {
        if ($null -eq $entry.$field -or [string]::IsNullOrWhiteSpace("$($entry.$field)")) {
            throw "Trigger policy entry '$($entry.id)' missing required field: $field"
        }
    }

    $metricName = "$($entry.metric)"
    if (-not $metrics.Contains($metricName)) {
        throw "Trigger policy entry '$($entry.id)' references unknown metric: $metricName"
    }

    [double]$actual = [double]$metrics[$metricName]
    [double]$targetMax = [double]$entry.targetMax
    [double]$allowedMax = [double]$entry.allowedMax
    $expiresOn = [datetime]::ParseExact("$($entry.expiry)", 'yyyy-MM-dd', [System.Globalization.CultureInfo]::InvariantCulture)
    $isExpired = $today -gt $expiresOn.Date

    if ($isExpired) {
        $policyViolations.Add("Trigger policy expired: id=$($entry.id) expiry=$($entry.expiry) owner=$($entry.owner) issue=$($entry.issue)")
    }

    if ($actual -gt $allowedMax) {
        $policyViolations.Add("Trigger policy guardrail exceeded: id=$($entry.id) metric=$metricName actual=$actual allowedMax=$allowedMax")
    }

    $entryEvaluations.Add([ordered]@{
            id         = "$($entry.id)"
            metric     = $metricName
            actual     = $actual
            targetMax  = $targetMax
            allowedMax = $allowedMax
            owner      = "$($entry.owner)"
            issue      = "$($entry.issue)"
            expiry     = "$($entry.expiry)"
            expired    = $isExpired
        }) | Out-Null
}

$effectiveEnforce = [bool]($Enforce -or $policyMode -eq 'enforce')

$report = [ordered]@{
    schema                                   = 'trigger_audit_report_v1'
    generatedAt                              = (Get-Date -Format 'o')
    enforce                                  = $effectiveEnforce
    astEvidenceRequired                      = [bool]$RequireAstEvidence
    policyMode                               = $policyMode
    policyPath                               = $resolvedPolicyPath
    activeDspMetricSource                    = $activeDspMetricSource
    retireFacadeMetricSource                 = $retireFacadeMetricSource
    retireFacadeRuntimeExecutionMetricSource = $retireFacadeRuntimeExecutionMetricSource
    runtimeExecutionViewMetricSource         = $runtimeExecutionViewMetricSource
    fadingOutDspMetricSource                 = $fadingOutDspMetricSource
    legacyDirectObserveMetricSource          = $legacyDirectObserveMetricSource
    metrics                                  = $metrics
    policyEvaluations                        = $entryEvaluations
    policyViolations                         = $policyViolations
    triggerTargets                           = [ordered]@{
        activeDspDeletionStart          = 0
        fadingOutDspDeletionStart       = 0
        retireFacadeRemovalStart        = 0
        runtimeExecutionViewConvergence = 0
        observeShimRemovalStart         = [ordered]@{
            runtimeExecutionViewUsageCountMaximum = 0
            legacyDirectObserveUsageCountMaximum  = 0
        }
    }
}

$reportJson = $report | ConvertTo-Json -Depth 8
Set-Content -LiteralPath $reportPath -Value $reportJson -Encoding UTF8

Write-Host "[INFO] trigger audit report written: $reportPath"
Write-Host "[INFO] activeDSP refs=$activeDspRefCount (source=$activeDspMetricSource raw=$activeDspRawRefCount) fadingOutDSP writes=$fadingOutDspWriteCount (source=$fadingOutDspMetricSource) retireFacade deps=$retireFacadeDirectDependencyCount (source=$retireFacadeMetricSource raw=$retireFacadeRawDependencyCount) retireFacadeRuntimeExec=$retireFacadeRuntimeExecutionCount (source=$retireFacadeRuntimeExecutionMetricSource) runtimeExecutionViewUsage=$runtimeExecutionViewUsageCount (source=$runtimeExecutionViewMetricSource) legacyDirectObserveUsage=$legacyDirectObserveUsageCount (source=$legacyDirectObserveMetricSource raw=$legacyDirectObserveRawCount)"

if ($policyViolations.Count -gt 0) {
    foreach ($violation in $policyViolations) {
        Write-Host "[ERROR] $violation"
    }
    throw "Trigger policy violations detected. count=$($policyViolations.Count)"
}

if ($effectiveEnforce) {
    foreach ($evaluation in $entryEvaluations) {
        if ([double]$evaluation.actual -gt [double]$evaluation.targetMax) {
            throw "Trigger enforce failed: id=$($evaluation.id) metric=$($evaluation.metric) actual=$($evaluation.actual) targetMax=$($evaluation.targetMax)"
        }
    }
}

Write-Host '[PASS] trigger audit completed'
