param(
    [string]$RegistryPath = '.github/isr-metric-governance.json'
)

$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$evidenceDir = Join-Path $repoRoot 'evidence'
$reportPath = Join-Path $evidenceDir 'metric_governance_report.json'
$resolvedRegistryPath = if ([System.IO.Path]::IsPathRooted($RegistryPath)) { $RegistryPath } else { Join-Path $repoRoot $RegistryPath }

if (-not (Test-Path $resolvedRegistryPath)) {
    throw "Missing metric governance registry: $resolvedRegistryPath"
}
if (-not (Test-Path $evidenceDir)) {
    New-Item -Path $evidenceDir -ItemType Directory | Out-Null
}

$registry = Get-Content -LiteralPath $resolvedRegistryPath -Raw -Encoding UTF8 | ConvertFrom-Json
if ($registry.schema -ne 'metric_governance_v1') {
    throw "Unexpected metric governance schema: $($registry.schema)"
}

foreach ($field in @('owner', 'issue', 'rationale', 'expiry')) {
    if ($null -eq $registry.$field -or [string]::IsNullOrWhiteSpace("$($registry.$field)")) {
        throw "metric governance registry missing required field: $field"
    }
}

$registryExpiry = [datetime]::ParseExact("$($registry.expiry)", 'yyyy-MM-dd', [System.Globalization.CultureInfo]::InvariantCulture)
if ((Get-Date).Date -gt $registryExpiry.Date) {
    throw "metric governance registry expired: expiry=$($registry.expiry) owner=$($registry.owner) issue=$($registry.issue)"
}

if (-not $registry.metrics -or $registry.metrics.Count -eq 0) {
    throw 'metric governance entries must be non-empty'
}

if ($null -eq $registry.normalizationPolicy) {
    throw 'metric governance registry missing normalizationPolicy object'
}

$normalizationPolicy = $registry.normalizationPolicy
if (-not [bool]$normalizationPolicy.enabled) {
    throw 'metric governance normalizationPolicy.enabled must be true'
}

$baselineWindowMinutes = [int]("$($normalizationPolicy.baselineWindowMinutes)")
if ($baselineWindowMinutes -le 0) {
    throw "metric governance normalizationPolicy.baselineWindowMinutes must be > 0 but was '$($normalizationPolicy.baselineWindowMinutes)'"
}

if (-not [bool]$normalizationPolicy.cpuThermalOsNormalization) {
    throw 'metric governance normalizationPolicy.cpuThermalOsNormalization must be true'
}

if ([string]::IsNullOrWhiteSpace("$($normalizationPolicy.bucketBy)")) {
    throw 'metric governance normalizationPolicy.bucketBy must be non-empty'
}

if ([string]::IsNullOrWhiteSpace("$($normalizationPolicy.issue)")) {
    throw 'metric governance normalizationPolicy.issue must be non-empty'
}

if ($null -eq $normalizationPolicy.strictModeRequireAllMetrics) {
    throw 'metric governance normalizationPolicy.strictModeRequireAllMetrics must be set'
}

$requiredMetricIds = @('xrunDelta', 'callbackJitter', 'retireLatency', 'crossfadePeak')
$today = (Get-Date).Date
$violations = New-Object System.Collections.Generic.List[string]
$evaluations = New-Object System.Collections.Generic.List[object]

foreach ($requiredId in $requiredMetricIds) {
    $entry = $registry.metrics | Where-Object { $_.id -eq $requiredId } | Select-Object -First 1
    if ($null -eq $entry) {
        $violations.Add("Missing required metric governance entry: id=$requiredId")
        continue
    }

    foreach ($field in @('id', 'blocking', 'owner', 'retention', 'normalization', 'threshold', 'action', 'issue', 'expiry')) {
        if ($null -eq $entry.$field -or [string]::IsNullOrWhiteSpace("$($entry.$field)")) {
            $violations.Add("Metric governance entry missing required field: id=$requiredId field=$field")
        }
    }

    if ("$($entry.normalization)" -ne 'baselineWindowNormalized') {
        $violations.Add("Metric governance entry has invalid normalization mode: id=$requiredId normalization=$($entry.normalization)")
    }

    if (@('yes', 'no') -notcontains "$($entry.blocking)") {
        $violations.Add("Metric governance entry has invalid blocking flag: id=$requiredId blocking=$($entry.blocking)")
    }

    $expiry = [datetime]::ParseExact("$($entry.expiry)", 'yyyy-MM-dd', [System.Globalization.CultureInfo]::InvariantCulture)
    $expired = $today -gt $expiry.Date
    if ($expired) {
        $violations.Add("Expired metric governance entry: id=$requiredId expiry=$($entry.expiry) owner=$($entry.owner) issue=$($entry.issue)")
    }

    $evaluations.Add([ordered]@{
            id            = "$($entry.id)"
            blocking      = "$($entry.blocking)"
            owner         = "$($entry.owner)"
            retention     = "$($entry.retention)"
            normalization = "$($entry.normalization)"
            threshold     = "$($entry.threshold)"
            action        = "$($entry.action)"
            issue         = "$($entry.issue)"
            expiry        = "$($entry.expiry)"
            expired       = $expired
        }) | Out-Null
}

$report = [ordered]@{
    schema              = 'metric_governance_report_v2'
    generatedAt         = (Get-Date -Format 'o')
    registryPath        = $resolvedRegistryPath
    registryOwner       = "$($registry.owner)"
    registryIssue       = "$($registry.issue)"
    registryExpiry      = "$($registry.expiry)"
    normalizationPolicy = [ordered]@{
        enabled                     = [bool]$normalizationPolicy.enabled
        baselineWindowMinutes       = $baselineWindowMinutes
        cpuThermalOsNormalization   = [bool]$normalizationPolicy.cpuThermalOsNormalization
        bucketBy                    = "$($normalizationPolicy.bucketBy)"
        strictModeRequireAllMetrics = [bool]$normalizationPolicy.strictModeRequireAllMetrics
        issue                       = "$($normalizationPolicy.issue)"
    }
    requiredMetrics     = $requiredMetricIds
    metrics             = $evaluations
    violations          = $violations
}

$reportJson = $report | ConvertTo-Json -Depth 8
Set-Content -LiteralPath $reportPath -Value $reportJson -Encoding UTF8
Write-Host "[INFO] metric governance report written: $reportPath"

if ($violations.Count -gt 0) {
    foreach ($violation in $violations) {
        Write-Host "[ERROR] $violation"
    }
    throw "Metric governance violations detected. count=$($violations.Count)"
}

Write-Host '[PASS] metric governance gate verified'
