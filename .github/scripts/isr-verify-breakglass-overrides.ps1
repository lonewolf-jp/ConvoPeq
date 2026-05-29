$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$registryPath = Join-Path $repoRoot '.github\isr-breakglass-overrides.json'
$evidenceDir = Join-Path $repoRoot 'evidence'
$reportPath = Join-Path $evidenceDir 'breakglass_overrides_report.json'

if (-not (Test-Path -LiteralPath $evidenceDir)) {
    New-Item -ItemType Directory -Path $evidenceDir -Force | Out-Null
}

if (-not (Test-Path -LiteralPath $registryPath)) {
    throw "Missing BreakGlass override registry: $registryPath"
}

$registry = Get-Content -LiteralPath $registryPath -Raw -Encoding UTF8 | ConvertFrom-Json
$violations = New-Object 'System.Collections.Generic.List[string]'

if ("$($registry.schema)" -ne 'isr_breakglass_overrides_v1') {
    $violations.Add("BreakGlass registry schema mismatch: expected=isr_breakglass_overrides_v1 actual=$($registry.schema)")
}

foreach ($field in @('owner', 'issue', 'rationale', 'expiry', 'entries')) {
    if ($null -eq $registry.$field) {
        $violations.Add("BreakGlass registry missing required field: $field")
    }
}

try {
    $registryExpiry = [datetime]::ParseExact("$($registry.expiry)", 'yyyy-MM-dd', [System.Globalization.CultureInfo]::InvariantCulture)
    if ((Get-Date).Date -gt $registryExpiry.Date) {
        $violations.Add("BreakGlass registry expired: expiry=$($registry.expiry)")
    }
}
catch {
    $violations.Add("BreakGlass registry has invalid expiry format: expiry=$($registry.expiry)")
}

$entries = @($registry.entries)
$seenIds = New-Object 'System.Collections.Generic.HashSet[string]'
$activeEntryCount = 0
$activeEntryDiagnostics = New-Object 'System.Collections.Generic.List[object]'

foreach ($entry in $entries) {
    foreach ($field in @('id', 'owner', 'reason', 'expiration', 'rollback_plan', 'approval')) {
        if ($null -eq $entry.$field -or [string]::IsNullOrWhiteSpace("$($entry.$field)")) {
            $violations.Add("BreakGlass entry missing required field: id=$($entry.id) field=$field")
        }
    }

    $entryId = "$($entry.id)"
    if (-not [string]::IsNullOrWhiteSpace($entryId)) {
        if (-not $seenIds.Add($entryId)) {
            $violations.Add("BreakGlass entry duplicate id detected: id=$entryId")
        }
    }

    try {
        $expiration = [datetime]::ParseExact("$($entry.expiration)", 'yyyy-MM-dd', [System.Globalization.CultureInfo]::InvariantCulture)
        $isActive = (Get-Date).Date -le $expiration.Date
        if ((Get-Date).Date -gt $expiration.Date) {
            $violations.Add("BreakGlass entry expired: id=$entryId expiration=$($entry.expiration)")
        }
        else {
            $activeEntryCount++

            $soakEvidenceRaw = "$($entry.soak_evidence)"
            $soakTierRaw = "$($entry.soak_tier)"
            if ([string]::IsNullOrWhiteSpace($soakEvidenceRaw)) {
                $violations.Add("BreakGlass active entry missing soak evidence field: id=$entryId field=soak_evidence")
            }

            if ([string]::IsNullOrWhiteSpace($soakTierRaw)) {
                $violations.Add("BreakGlass active entry missing soak evidence field: id=$entryId field=soak_tier")
            }

            $resolvedSoakEvidencePath = $null
            $soakEvidenceExists = $false
            if (-not [string]::IsNullOrWhiteSpace($soakEvidenceRaw)) {
                $resolvedSoakEvidencePath = if ([System.IO.Path]::IsPathRooted($soakEvidenceRaw)) {
                    [System.IO.Path]::GetFullPath($soakEvidenceRaw)
                }
                else {
                    [System.IO.Path]::GetFullPath((Join-Path $repoRoot $soakEvidenceRaw))
                }

                $soakEvidenceExists = Test-Path -LiteralPath $resolvedSoakEvidencePath
                if (-not $soakEvidenceExists) {
                    $violations.Add("BreakGlass active entry soak evidence file not found: id=$entryId soak_evidence=$soakEvidenceRaw")
                }
            }

            $activeEntryDiagnostics.Add([ordered]@{
                    id = $entryId
                    expiration = "$($entry.expiration)"
                    isActive = $isActive
                    soakTier = $soakTierRaw
                    soakEvidence = $soakEvidenceRaw
                    soakEvidenceResolvedPath = $resolvedSoakEvidencePath
                    soakEvidenceExists = $soakEvidenceExists
                }) | Out-Null
        }
    }
    catch {
        $violations.Add("BreakGlass entry has invalid expiration format: id=$entryId expiration=$($entry.expiration)")
    }

    if ($entry.PSObject.Properties.Name -contains 'persistent' -and [bool]$entry.persistent) {
        $violations.Add("BreakGlass persistent override is forbidden: id=$entryId")
    }

    if ($entry.PSObject.Properties.Name -contains 'releasePersistent' -and [bool]$entry.releasePersistent) {
        $violations.Add("BreakGlass release persistent override is forbidden: id=$entryId")
    }
}

$report = [ordered]@{
    schema = 'breakglass_overrides_report_v1'
    generatedAt = (Get-Date -Format 'o')
    registryPath = $registryPath
    summary = [ordered]@{
        totalEntries = $entries.Count
        activeEntries = $activeEntryCount
    }
    activeEntryDiagnostics = $activeEntryDiagnostics
    violations = @($violations)
    ready = ($violations.Count -eq 0)
}

$report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $reportPath -Encoding UTF8
Write-Host "[INFO] breakglass override report written: $reportPath"

if ($violations.Count -gt 0) {
    foreach ($violation in $violations) {
        Write-Host "[ERROR] $violation"
    }
    throw "BreakGlass override verification failed. violations=$($violations.Count)"
}

Write-Host '[PASS] BreakGlass override verification passed'
