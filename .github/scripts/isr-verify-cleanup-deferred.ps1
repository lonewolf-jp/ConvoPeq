param(
    [string]$RegistryPath = '.github/isr-cleanup-deferred.json',
    [string]$PruneReportPath = 'evidence/cleanup_deferred_prune_report.json'
)

$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$evidenceDir = Join-Path $repoRoot 'evidence'
$reportPath = Join-Path $evidenceDir 'cleanup_deferred_report.json'
$resolvedRegistryPath = if ([System.IO.Path]::IsPathRooted($RegistryPath)) { $RegistryPath } else { Join-Path $repoRoot $RegistryPath }
$resolvedPruneReportPath = if ([System.IO.Path]::IsPathRooted($PruneReportPath)) { $PruneReportPath } else { Join-Path $repoRoot $PruneReportPath }

if (-not (Test-Path $resolvedRegistryPath)) {
    throw "Missing cleanup deferred registry: $resolvedRegistryPath"
}
if (-not (Test-Path $resolvedPruneReportPath)) {
    throw "Missing cleanup deferred prune report: $resolvedPruneReportPath"
}
if (-not (Test-Path $evidenceDir)) {
    New-Item -Path $evidenceDir -ItemType Directory | Out-Null
}

$registry = Get-Content -LiteralPath $resolvedRegistryPath -Raw -Encoding UTF8 | ConvertFrom-Json
$pruneReport = Get-Content -LiteralPath $resolvedPruneReportPath -Raw -Encoding UTF8 | ConvertFrom-Json
if ($registry.schema -ne 'cleanup_deferred_registry_v1') {
    throw "Unexpected cleanup deferred schema: $($registry.schema)"
}
if ($pruneReport.schema -ne 'cleanup_deferred_prune_report_v1') {
    throw "Unexpected cleanup deferred prune report schema: $($pruneReport.schema)"
}

foreach ($field in @('apply', 'readyCount', 'blockedCount', 'prunedCount', 'remainingCount')) {
    if ($null -eq $pruneReport.$field) {
        throw "cleanup deferred prune report missing required field: $field"
    }
}

if ($null -eq $pruneReport.violations) {
    throw 'cleanup deferred prune report missing violations field'
}

if (@($pruneReport.violations).Count -gt 0) {
    throw "cleanup deferred prune report contains violations. count=$(@($pruneReport.violations).Count)"
}

foreach ($field in @('owner', 'issue', 'rationale', 'expiry')) {
    if ($null -eq $registry.$field -or [string]::IsNullOrWhiteSpace("$($registry.$field)")) {
        throw "cleanup deferred registry missing required field: $field"
    }
}

$registryExpiry = [datetime]::ParseExact("$($registry.expiry)", 'yyyy-MM-dd', [System.Globalization.CultureInfo]::InvariantCulture)
if ((Get-Date).Date -gt $registryExpiry.Date) {
    throw "cleanup deferred registry expired: expiry=$($registry.expiry) owner=$($registry.owner) issue=$($registry.issue)"
}

$entryCount = if ($null -eq $registry.entries) { 0 } else { [int]$registry.entries.Count }

if ([int]$pruneReport.remainingCount -ne $entryCount) {
    throw "cleanup deferred prune report remainingCount mismatch: remainingCount=$($pruneReport.remainingCount) registryEntryCount=$entryCount"
}

if (-not [bool]$pruneReport.apply -and [int]$pruneReport.prunedCount -ne 0) {
    throw "cleanup deferred prune report consistency error: apply=false requires prunedCount=0 but was $($pruneReport.prunedCount)"
}

$today = (Get-Date).Date
$violations = New-Object System.Collections.Generic.List[string]
$evaluations = New-Object System.Collections.Generic.List[object]

foreach ($entry in ($registry.entries | Where-Object { $null -ne $_ })) {
    foreach ($field in @('id', 'trigger', 'owner', 'issue', 'rationale', 'expiry')) {
        if ($null -eq $entry.$field -or [string]::IsNullOrWhiteSpace("$($entry.$field)")) {
            throw "cleanup deferred entry '$($entry.id)' missing required field: $field"
        }
    }

    $expiry = [datetime]::ParseExact("$($entry.expiry)", 'yyyy-MM-dd', [System.Globalization.CultureInfo]::InvariantCulture)
    $expired = $today -gt $expiry.Date
    if ($expired) {
        $violations.Add("Expired cleanup deferred entry: id=$($entry.id) trigger=$($entry.trigger) owner=$($entry.owner) issue=$($entry.issue) expiry=$($entry.expiry)")
    }

    $evaluations.Add([ordered]@{
            id      = "$($entry.id)"
            trigger = "$($entry.trigger)"
            owner   = "$($entry.owner)"
            issue   = "$($entry.issue)"
            expiry  = "$($entry.expiry)"
            expired = $expired
        }) | Out-Null
}

$report = [ordered]@{
    schema          = 'cleanup_deferred_report_v1'
    generatedAt     = (Get-Date -Format 'o')
    registryPath    = $resolvedRegistryPath
    pruneReportPath = $resolvedPruneReportPath
    registryOwner   = "$($registry.owner)"
    registryIssue   = "$($registry.issue)"
    registryExpiry  = "$($registry.expiry)"
    entryCount      = $entryCount
    pruneSummary    = [ordered]@{
        apply          = [bool]$pruneReport.apply
        readyCount     = [int]$pruneReport.readyCount
        blockedCount   = [int]$pruneReport.blockedCount
        prunedCount    = [int]$pruneReport.prunedCount
        remainingCount = [int]$pruneReport.remainingCount
    }
    entries         = $evaluations
    violations      = $violations
}

$reportJson = $report | ConvertTo-Json -Depth 8
Set-Content -LiteralPath $reportPath -Value $reportJson -Encoding UTF8
Write-Host "[INFO] cleanup deferred report written: $reportPath"

if ($violations.Count -gt 0) {
    foreach ($violation in $violations) {
        Write-Host "[ERROR] $violation"
    }
    throw "cleanup deferred registry violations detected. count=$($violations.Count)"
}

if ($entryCount -eq 0) {
    Write-Host '[INFO] cleanup deferred registry is empty (all deferred cleanup completed)'
}

Write-Host '[PASS] cleanup deferred registry gate verified'
