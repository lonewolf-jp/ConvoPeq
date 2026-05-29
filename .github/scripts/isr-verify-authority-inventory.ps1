param(
    [string]$RepoRoot = (Join-Path $PSScriptRoot '..\..'),
    [string]$InventoryDir = 'storage\isr_inventory',
    [bool]$Regenerate = $true
)

$ErrorActionPreference = 'Stop'

$repoRootResolved = [System.IO.Path]::GetFullPath($RepoRoot)
$inventoryDirResolved = Join-Path $repoRootResolved $InventoryDir
$evidenceDir = Join-Path $repoRootResolved 'evidence'
$reportPath = Join-Path $evidenceDir 'authority_inventory_report.json'
$legacyManifestPath = Join-Path $repoRootResolved '.github\isr-legacy-temporary.json'
$generateScriptPath = Join-Path $repoRootResolved '.github\scripts\isr-generate-authority-inventory.ps1'

if (-not (Test-Path -LiteralPath $evidenceDir)) {
    New-Item -ItemType Directory -Path $evidenceDir -Force | Out-Null
}

if (-not (Test-Path -LiteralPath $generateScriptPath)) {
    throw "Missing generator script: $generateScriptPath"
}

if ($Regenerate -or -not (Test-Path -LiteralPath (Join-Path $inventoryDirResolved 'post_authority_inventory.json'))) {
    & $generateScriptPath -RepoRoot $repoRootResolved -OutputDir $InventoryDir
}

$currentInventoryPath = Join-Path $inventoryDirResolved 'current_authority_inventory.json'
$postInventoryPath = Join-Path $inventoryDirResolved 'post_authority_inventory.json'
$diffInventoryPath = Join-Path $inventoryDirResolved 'inventory_diff_report.json'

foreach ($requiredPath in @($legacyManifestPath, $currentInventoryPath, $postInventoryPath, $diffInventoryPath)) {
    if (-not (Test-Path -LiteralPath $requiredPath)) {
        throw "Missing required inventory artifact: $requiredPath"
    }
}

$legacyManifest = Get-Content -LiteralPath $legacyManifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
$currentInventory = Get-Content -LiteralPath $currentInventoryPath -Raw -Encoding UTF8 | ConvertFrom-Json
$postInventory = Get-Content -LiteralPath $postInventoryPath -Raw -Encoding UTF8 | ConvertFrom-Json
$diffInventory = Get-Content -LiteralPath $diffInventoryPath -Raw -Encoding UTF8 | ConvertFrom-Json

$violations = New-Object System.Collections.Generic.List[string]
$warnings = New-Object System.Collections.Generic.List[string]

if ("$($legacyManifest.schema)" -ne 'isr_legacy_temporary_manifest_v1') {
    $violations.Add("Legacy manifest schema mismatch: expected=isr_legacy_temporary_manifest_v1 actual=$($legacyManifest.schema)")
}

foreach ($field in @('symbol', 'owner', 'replacement_authority', 'removal_phase', 'deadline', 'scope', 'entries')) {
    if ($null -eq $legacyManifest.$field) {
        $violations.Add("Legacy manifest missing required field: $field")
    }
}

foreach ($field in @('issue', 'expiry')) {
    if ($null -eq $legacyManifest.$field -or [string]::IsNullOrWhiteSpace("$($legacyManifest.$field)")) {
        $violations.Add("Legacy manifest missing required governance field: $field")
    }
}

if ("$($legacyManifest.removal_phase)" -ne 'Phase 3') {
    $violations.Add("Legacy manifest removal_phase must be Phase 3 (actual=$($legacyManifest.removal_phase))")
}

if ("$($legacyManifest.replacement_authority)" -ne 'Authoritative') {
    $violations.Add("Legacy manifest replacement_authority must be Authoritative (actual=$($legacyManifest.replacement_authority))")
}

try {
    $manifestDeadline = [datetime]::ParseExact("$($legacyManifest.deadline)", 'yyyy-MM-dd', [System.Globalization.CultureInfo]::InvariantCulture)
    if ((Get-Date).Date -gt $manifestDeadline.Date) {
        $violations.Add("Legacy manifest deadline expired: deadline=$($legacyManifest.deadline)")
    }
}
catch {
    $violations.Add("Legacy manifest has invalid deadline format: deadline=$($legacyManifest.deadline)")
}

try {
    $manifestExpiry = [datetime]::ParseExact("$($legacyManifest.expiry)", 'yyyy-MM-dd', [System.Globalization.CultureInfo]::InvariantCulture)
    if ((Get-Date).Date -gt $manifestExpiry.Date) {
        $violations.Add("Legacy manifest expiry exceeded: expiry=$($legacyManifest.expiry)")
    }
}
catch {
    $violations.Add("Legacy manifest has invalid expiry format: expiry=$($legacyManifest.expiry)")
}

if ("$($postInventory.schema)" -ne 'authority_inventory_v1') {
    $violations.Add("Post inventory schema mismatch: expected=authority_inventory_v1 actual=$($postInventory.schema)")
}

if ("$($currentInventory.schema)" -ne 'authority_inventory_v1') {
    $violations.Add("Current inventory schema mismatch: expected=authority_inventory_v1 actual=$($currentInventory.schema)")
}

if ("$($diffInventory.schema)" -ne 'authority_inventory_diff_report_v1') {
    $violations.Add("Diff inventory schema mismatch: expected=authority_inventory_diff_report_v1 actual=$($diffInventory.schema)")
}

$legacyEntries = @($legacyManifest.entries)
$legacyByState = @{}
$legacyDuplicates = New-Object System.Collections.Generic.List[string]

foreach ($entry in $legacyEntries) {
    foreach ($requiredEntryField in @('state', 'owner', 'replacement_authority', 'removal_phase', 'deadline', 'scope')) {
        if ($null -eq $entry.$requiredEntryField -or [string]::IsNullOrWhiteSpace("$($entry.$requiredEntryField)")) {
            $violations.Add("Legacy manifest entry missing required field: $requiredEntryField")
        }
    }

    $state = "$($entry.state)"
    if ([string]::IsNullOrWhiteSpace($state)) {
        continue
    }

    if ($legacyByState.ContainsKey($state)) {
        $legacyDuplicates.Add($state)
        continue
    }

    $legacyByState[$state] = $entry

    try {
        $entryExpiry = [datetime]::ParseExact("$($entry.deadline)", 'yyyy-MM-dd', [System.Globalization.CultureInfo]::InvariantCulture)
        if ((Get-Date).Date -gt $entryExpiry.Date) {
            $violations.Add("Legacy manifest entry expired: state=$state deadline=$($entry.deadline)")
        }
    }
    catch {
        $violations.Add("Legacy manifest entry has invalid deadline format: state=$state deadline=$($entry.deadline)")
    }
}

foreach ($duplicateState in $legacyDuplicates) {
    $violations.Add("Legacy manifest duplicate state entry detected: state=$duplicateState")
}

$postEntries = @($postInventory.entries)
$legacyInventoryEntries = @($postEntries | Where-Object { "$($_.authority_class)" -eq 'LegacyTemporary' })

$nonSinglePublicationEntries = @($postEntries | Where-Object { "$($_.publication_path)" -ne 'publish(RuntimeWorld*)' })
foreach ($entry in $nonSinglePublicationEntries) {
    $violations.Add("Publication path drift: single publication contract violated: state=$($entry.state) publication_path=$($entry.publication_path)")
}

$nonRuntimeWorldObserveEntries = @($postEntries | Where-Object { "$($_.observe_path)" -ne 'RuntimeWorld' })
foreach ($entry in $nonRuntimeWorldObserveEntries) {
    $violations.Add("Observe path drift: RuntimeWorld only contract violated: state=$($entry.state) observe_path=$($entry.observe_path)")
}

$retirementOwners = @($postEntries | ForEach-Object { "$($_.retirement_owner)" } | Where-Object { -not [string]::IsNullOrWhiteSpace($_) } | Select-Object -Unique)
if ($retirementOwners.Count -gt 1) {
    $violations.Add("Retirement owner singularization violated: owners=$($retirementOwners -join ',')")
}

$stateCounts = @{}
foreach ($entry in $postEntries) {
    $stateName = "$($entry.state)"
    if (-not $stateCounts.ContainsKey($stateName)) {
        $stateCounts[$stateName] = 0
    }
    $stateCounts[$stateName] += 1
}

foreach ($stateName in $stateCounts.Keys) {
    if ($stateCounts[$stateName] -gt 2) {
        $violations.Add("Semantic duplication budget violated (>2): state=$stateName count=$($stateCounts[$stateName])")
    }
}

if ($null -ne $diffInventory.summary) {
    $addedCount = [int]$diffInventory.summary.addedCount
    if ($addedCount -gt 0) {
        $violations.Add("Authority source growth detected: addedCount=$addedCount")
    }

    if ($diffInventory.summary.PSObject.Properties.Name -contains 'observePathChangedCount') {
        $observePathChangedCount = [int]$diffInventory.summary.observePathChangedCount
        if ($observePathChangedCount -gt 0) {
            $violations.Add("Observe path growth/drift detected: observePathChangedCount=$observePathChangedCount")
        }
    }

    if ($diffInventory.summary.PSObject.Properties.Name -contains 'retirementOwnerChangedCount') {
        $retirementOwnerChangedCount = [int]$diffInventory.summary.retirementOwnerChangedCount
        if ($retirementOwnerChangedCount -gt 0) {
            $violations.Add("Retirement owner drift detected: retirementOwnerChangedCount=$retirementOwnerChangedCount")
        }
    }
}

$missingManifestCoverage = New-Object System.Collections.Generic.List[string]
foreach ($legacyInventoryEntry in $legacyInventoryEntries) {
    $state = "$($legacyInventoryEntry.state)"
    if (-not $legacyByState.ContainsKey($state)) {
        $missingManifestCoverage.Add($state)
    }
}

foreach ($state in $missingManifestCoverage) {
    $violations.Add("LegacyTemporary state is not tracked by manifest: state=$state")
}

$staleManifestEntries = New-Object System.Collections.Generic.List[string]
$legacyInventoryStateSet = @{}
foreach ($legacyInventoryEntry in $legacyInventoryEntries) {
    $legacyInventoryStateSet["$($legacyInventoryEntry.state)"] = $true
}

foreach ($manifestState in $legacyByState.Keys) {
    if (-not $legacyInventoryStateSet.ContainsKey($manifestState)) {
        $staleManifestEntries.Add($manifestState)
    }
}

foreach ($state in $staleManifestEntries) {
    $violations.Add("Legacy manifest entry is stale (state not present in inventory): state=$state")
}

$report = [ordered]@{
    schema                       = 'authority_inventory_report_v1'
    generatedAt                  = (Get-Date -Format 'o')
    inventoryDir                 = $inventoryDirResolved
    artifacts                    = [ordered]@{
        current        = $currentInventoryPath
        post           = $postInventoryPath
        diff           = $diffInventoryPath
        legacyManifest = $legacyManifestPath
    }
    summary                      = [ordered]@{
        postTotal                       = $postEntries.Count
        legacyTemporaryTotal            = $legacyInventoryEntries.Count
        legacyManifestTotal             = $legacyEntries.Count
        missingManifestCoverageCount    = $missingManifestCoverage.Count
        staleManifestEntryCount         = $staleManifestEntries.Count
        nonSinglePublicationCount       = $nonSinglePublicationEntries.Count
        nonRuntimeWorldObserveCount     = $nonRuntimeWorldObserveEntries.Count
        retirementOwnerCount            = $retirementOwners.Count
        diffAddedCount                  = [int]$diffInventory.summary.addedCount
        diffRemovedCount                = [int]$diffInventory.summary.removedCount
        diffClassChangedCount           = [int]$diffInventory.summary.classChangedCount
        diffObservePathChangedCount     = if ($diffInventory.summary.PSObject.Properties.Name -contains 'observePathChangedCount') { [int]$diffInventory.summary.observePathChangedCount } else { 0 }
        diffRetirementOwnerChangedCount = if ($diffInventory.summary.PSObject.Properties.Name -contains 'retirementOwnerChangedCount') { [int]$diffInventory.summary.retirementOwnerChangedCount } else { 0 }
    }
    legacyTemporaryStates        = @($legacyInventoryEntries | ForEach-Object { "$($_.state)" })
    nonSinglePublicationStates   = @($nonSinglePublicationEntries | ForEach-Object { "$($_.state)" })
    nonRuntimeWorldObserveStates = @($nonRuntimeWorldObserveEntries | ForEach-Object { "$($_.state)" })
    retirementOwners             = @($retirementOwners)
    missingManifestCoverage      = @($missingManifestCoverage)
    staleManifestEntries         = @($staleManifestEntries)
    warnings                     = @($warnings)
    violations                   = @($violations)
}

$report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $reportPath -Encoding UTF8
Write-Host "[INFO] authority inventory report written: $reportPath"

foreach ($warning in $warnings) {
    Write-Host "[WARN] $warning"
}

if ($violations.Count -gt 0) {
    foreach ($violation in $violations) {
        Write-Host "[ERROR] $violation"
    }
    throw "Authority inventory verification failed. violations=$($violations.Count)"
}

Write-Host '[PASS] authority inventory verification passed'
