param(
    [ValidateSet('smoke', 'standard', 'exhaustive')]
    [string]$Tier = 'standard',
    [string]$DeclaredClass = '',
    [string]$OpenedAt = '',
    [int]$SoakMinutes = 0,
    [switch]$RequireDeclaredClass,
    [string]$Now = '',
    [switch]$ReleaseWindow,
    [string]$GitRef = '',
    [string]$PolicyPath = (Join-Path $PSScriptRoot '..\isr-pr-sla-policy.json')
)

$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$policyFullPath = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\isr-pr-sla-policy.json'))
$evidencePath = Join-Path $repoRoot 'evidence\pr_sla_report.json'
$breakglassReportPath = Join-Path $repoRoot 'evidence\breakglass_overrides_report.json'
$safetyReportPath = Join-Path $repoRoot 'evidence\safety_regression_report.json'
$authorityInventoryReportPath = Join-Path $repoRoot 'evidence\authority_inventory_report.json'
$inventoryDiffReportPath = Join-Path $repoRoot 'storage\isr_inventory\inventory_diff_report.json'

if (-not (Test-Path -LiteralPath $policyFullPath)) {
    throw "Missing PR SLA policy: $policyFullPath"
}

$policy = Get-Content -LiteralPath $policyFullPath -Raw -Encoding UTF8 | ConvertFrom-Json
if ($policy.schema -ne 'isr_pr_sla_policy_v1') {
    throw "Unexpected PR SLA policy schema: $($policy.schema)"
}

if ($null -eq $policy.releaseRequiresExhaustive) {
    throw 'PR SLA policy missing required field: releaseRequiresExhaustive'
}

$classes = @($policy.classes)
if ($classes.Count -eq 0) {
    throw 'PR SLA policy classes must be non-empty'
}

$violations = @()

if ($SoakMinutes -lt 0) {
    $violations += "SoakMinutes must be non-negative: actual=$SoakMinutes"
}

function Test-InventoryDiffStructuralInvariant {
    param(
        [pscustomobject]$InventoryDiffReport
    )

    if ($null -eq $InventoryDiffReport -or $null -eq $InventoryDiffReport.summary) {
        return $false
    }

    $summary = $InventoryDiffReport.summary
    $isInvariant =
        ([int]$summary.addedCount -eq 0) -and
        ([int]$summary.removedCount -eq 0) -and
        ([int]$summary.classChangedCount -eq 0) -and
        ([int]$summary.observePathChangedCount -eq 0) -and
        ([int]$summary.publicationPathChangedCount -eq 0) -and
        ([int]$summary.retirementOwnerChangedCount -eq 0) -and
        ([int]$summary.ownerChangedCount -eq 0)

    return $isInvariant
}

function Resolve-RuntimeCodeChangeSignal {
    param(
        [string]$RepoRoot
    )

    $result = [ordered]@{
        detectionReady = $false
        detectionMode = 'none'
        runtimeChangedFiles = @()
        totalChangedFiles = 0
        detail = $null
    }

    $eventPath = $env:GITHUB_EVENT_PATH
    if ([string]::IsNullOrWhiteSpace($eventPath) -or -not (Test-Path -LiteralPath $eventPath)) {
        return [pscustomobject]$result
    }

    try {
        $eventPayload = Get-Content -LiteralPath $eventPath -Raw -Encoding UTF8 | ConvertFrom-Json
    }
    catch {
        $result.detail = "failed_to_parse_event_payload: $($_.Exception.Message)"
        return [pscustomobject]$result
    }

    $baseSha = "$($eventPayload.pull_request.base.sha)"
    $headSha = "$($eventPayload.pull_request.head.sha)"
    if ([string]::IsNullOrWhiteSpace($baseSha) -or [string]::IsNullOrWhiteSpace($headSha)) {
        $result.detail = 'pull_request base/head sha unavailable'
        return [pscustomobject]$result
    }

    try {
        $changedFiles = @(& git -C $RepoRoot diff --name-only "$baseSha..$headSha")
        if ($LASTEXITCODE -ne 0) {
            throw "git diff exited with code $LASTEXITCODE"
        }

        $normalized = @($changedFiles | Where-Object { -not [string]::IsNullOrWhiteSpace($_) } | ForEach-Object { ($_ -replace '\\', '/') })
        $runtimeChanged = @($normalized | Where-Object { $_ -match '^(src/|source/)' })

        $result.detectionReady = $true
        $result.detectionMode = 'pull_request_base_head_git_diff'
        $result.runtimeChangedFiles = $runtimeChanged
        $result.totalChangedFiles = $normalized.Count
        return [pscustomobject]$result
    }
    catch {
        $result.detail = "git_diff_failed: $($_.Exception.Message)"
        return [pscustomobject]$result
    }
}

function Add-BusinessDays {
    param(
        [datetime]$Start,
        [int]$BusinessDays
    )

    $current = $Start.Date
    $remaining = [Math]::Max(0, $BusinessDays)
    while ($remaining -gt 0) {
        $current = $current.AddDays(1)
        if ($current.DayOfWeek -ne [DayOfWeek]::Saturday -and $current.DayOfWeek -ne [DayOfWeek]::Sunday) {
            $remaining--
        }
    }

    return $current
}

function Resolve-DeclaredClassFromEventLabels {
    param(
        [string]$EventPath
    )

    if ([string]::IsNullOrWhiteSpace($EventPath) -or -not (Test-Path -LiteralPath $EventPath)) {
        return $null
    }

    try {
        $eventPayload = Get-Content -LiteralPath $EventPath -Raw -Encoding UTF8 | ConvertFrom-Json
        if ($null -eq $eventPayload.pull_request -or $null -eq $eventPayload.pull_request.labels) {
            return $null
        }

        $resolvedClasses = New-Object System.Collections.Generic.List[string]
        foreach ($label in @($eventPayload.pull_request.labels)) {
            $name = "$($label.name)"
            if ([string]::IsNullOrWhiteSpace($name)) {
                continue
            }

            if ($name -match '^isr-pr-class:(Class-[SABCD])$') {
                $resolvedClasses.Add($Matches[1]) | Out-Null
            }
            elseif ($name -match '^(Class-[SABCD])$') {
                $resolvedClasses.Add($Matches[1]) | Out-Null
            }
        }

        $unique = @($resolvedClasses | Select-Object -Unique)
        if ($unique.Count -eq 1) {
            return [pscustomobject]@{
                declaredClass = $unique[0]
                source = 'github-event-label'
            }
        }

        if ($unique.Count -gt 1) {
            return [pscustomobject]@{
                declaredClass = $null
                source = 'github-event-label-conflict'
                conflictLabels = $unique
            }
        }
    }
    catch {
        return [pscustomobject]@{
            declaredClass = $null
            source = 'github-event-parse-error'
            error = "$($_.Exception.Message)"
        }
    }

    return $null
}

function Resolve-OpenedAtFromEvent {
    param(
        [string]$EventPath
    )

    if ([string]::IsNullOrWhiteSpace($EventPath) -or -not (Test-Path -LiteralPath $EventPath)) {
        return $null
    }

    try {
        $eventPayload = Get-Content -LiteralPath $EventPath -Raw -Encoding UTF8 | ConvertFrom-Json
        $createdAtRaw = "$($eventPayload.pull_request.created_at)"
        if ([string]::IsNullOrWhiteSpace($createdAtRaw)) {
            return $null
        }

        return [datetime]::Parse($createdAtRaw)
    }
    catch {
        return $null
    }
}

function Resolve-EventHeadSha {
    param(
        [string]$EventPath
    )

    if ([string]::IsNullOrWhiteSpace($EventPath) -or -not (Test-Path -LiteralPath $EventPath)) {
        return $null
    }

    try {
        $eventPayload = Get-Content -LiteralPath $EventPath -Raw -Encoding UTF8 | ConvertFrom-Json
        $headSha = "$($eventPayload.pull_request.head.sha)"
        if ([string]::IsNullOrWhiteSpace($headSha)) {
            return $null
        }

        return $headSha
    }
    catch {
        return $null
    }
}

$requiredClassIds = @('Class-S', 'Class-A', 'Class-B', 'Class-C', 'Class-D')

foreach ($classId in $requiredClassIds) {
    if (-not (@($classes) | Where-Object { [string]$_.id -eq $classId })) {
        $violations += "Missing PR class definition: $classId"
    }
}

$declaredClass = $DeclaredClass
$declaredClassSource = 'explicit-argument'
if ([string]::IsNullOrWhiteSpace($declaredClass)) {
    $declaredClassSource = 'default'
    $resolvedFromLabels = Resolve-DeclaredClassFromEventLabels -EventPath $env:GITHUB_EVENT_PATH
    if ($null -ne $resolvedFromLabels -and -not [string]::IsNullOrWhiteSpace("$($resolvedFromLabels.declaredClass)")) {
        $declaredClass = "$($resolvedFromLabels.declaredClass)"
        $declaredClassSource = "$($resolvedFromLabels.source)"
    }

    if ([string]::IsNullOrWhiteSpace($declaredClass) -and $RequireDeclaredClass) {
        $violations += 'Declared PR class is required but missing (expected workflow input or PR label isr-pr-class:Class-*)'
    }

    if ([string]::IsNullOrWhiteSpace($declaredClass)) {
        $declaredClass = 'Class-A'
    }
}

if ($declaredClassSource -eq 'default' -and $RequireDeclaredClass) {
    $violations += "Declared PR class requirement fell back to default class: $declaredClass"
}

if (-not [string]::IsNullOrWhiteSpace($declaredClass) -and $declaredClassSource -eq 'default') {
    $declaredClass = 'Class-A'
}

$declaredPolicy = @($classes) | Where-Object { [string]$_.id -eq $declaredClass } | Select-Object -First 1
if ($null -eq $declaredPolicy) {
    $violations += "Unknown declared PR class: $declaredClass"
}
else {
    $tierOrder = @{ smoke = 0; standard = 1; exhaustive = 2 }
    if ($tierOrder[$Tier] -lt $tierOrder[[string]$declaredPolicy.minimumTier]) {
        $violations += "Tier $Tier does not satisfy minimum tier $($declaredPolicy.minimumTier) for $declaredClass"
    }
}

$requiredNotes = @()
if ($null -ne $declaredPolicy) {
    $requiredNotes = @($declaredPolicy.requiredNotes)
    if ($requiredNotes.Count -eq 0) {
        $requiresContractNote = ([bool]$declaredPolicy.runtimeCodeChangeZeroRequired) -or
                                ([bool]$declaredPolicy.inventoryDiffStructuralInvariantRequired)
        if ($requiresContractNote) {
            $violations += "PR class policy missing requiredNotes: class=$declaredClass"
        }
    }
}

$effectiveGitRef = $GitRef
if ([string]::IsNullOrWhiteSpace($effectiveGitRef)) {
    if (-not [string]::IsNullOrWhiteSpace($env:GITHUB_REF)) {
        $effectiveGitRef = $env:GITHUB_REF
    } elseif (-not [string]::IsNullOrWhiteSpace($env:BUILD_SOURCEBRANCH)) {
        $effectiveGitRef = $env:BUILD_SOURCEBRANCH
    }
}

$isReleaseRef = $false
if (-not [string]::IsNullOrWhiteSpace($effectiveGitRef)) {
    $isReleaseRef = [regex]::IsMatch($effectiveGitRef, '^refs/heads/release(/|$)|^refs/tags/')
}

$releaseWindowActive = [bool]$ReleaseWindow -or $isReleaseRef
if ([bool]$policy.releaseRequiresExhaustive -and $releaseWindowActive -and $Tier -ne 'exhaustive') {
    $violations += "Release window requires exhaustive tier for all classes: tier=$Tier gitRef=$effectiveGitRef"
}

$breakglassReport = $null
$breakglassActiveEntries = 0
$breakglassReportReady = $false
if (Test-Path -LiteralPath $breakglassReportPath) {
    try {
        $breakglassReport = Get-Content -LiteralPath $breakglassReportPath -Raw -Encoding UTF8 | ConvertFrom-Json
        if ("$($breakglassReport.schema)" -ne 'breakglass_overrides_report_v1') {
            $violations += "Unexpected breakglass report schema: $($breakglassReport.schema)"
        }
        else {
            $breakglassActiveEntries = [int]$breakglassReport.summary.activeEntries
            $breakglassReportReady = [bool]$breakglassReport.ready
        }
    }
    catch {
        $violations += "Failed to parse breakglass report: $($_.Exception.Message)"
    }
}

$authorityInventoryReport = $null
if (Test-Path -LiteralPath $authorityInventoryReportPath) {
    try {
        $authorityInventoryReport = Get-Content -LiteralPath $authorityInventoryReportPath -Raw -Encoding UTF8 | ConvertFrom-Json
        if ("$($authorityInventoryReport.schema)" -ne 'authority_inventory_report_v1') {
            $violations += "Unexpected authority inventory report schema: $($authorityInventoryReport.schema)"
            $authorityInventoryReport = $null
        }
    }
    catch {
        $violations += "Failed to parse authority inventory report: $($_.Exception.Message)"
    }
}

$inventoryDiffReport = $null
if (Test-Path -LiteralPath $inventoryDiffReportPath) {
    try {
        $inventoryDiffReport = Get-Content -LiteralPath $inventoryDiffReportPath -Raw -Encoding UTF8 | ConvertFrom-Json
        if ("$($inventoryDiffReport.schema)" -ne 'authority_inventory_diff_report_v1') {
            $violations += "Unexpected inventory diff report schema: $($inventoryDiffReport.schema)"
            $inventoryDiffReport = $null
        }
    }
    catch {
        $violations += "Failed to parse inventory diff report: $($_.Exception.Message)"
    }
}

$inventoryStructuralInvariant = Test-InventoryDiffStructuralInvariant -InventoryDiffReport $inventoryDiffReport
$runtimeCodeChangeSignal = Resolve-RuntimeCodeChangeSignal -RepoRoot $repoRoot
$runtimeCodeChangeZeroConfidence = 'inventory-fallback'
$runtimeCodeChangeZero = $inventoryStructuralInvariant
if ([bool]$runtimeCodeChangeSignal.detectionReady) {
    $runtimeCodeChangeZeroConfidence = 'git-diff-runtime-scan'
    $runtimeCodeChangeZero = (@($runtimeCodeChangeSignal.runtimeChangedFiles).Count -eq 0)
}

$safetyReport = $null
if (Test-Path -LiteralPath $safetyReportPath) {
    try {
        $safetyReport = Get-Content -LiteralPath $safetyReportPath -Raw -Encoding UTF8 | ConvertFrom-Json
        if ("$($safetyReport.schema)" -ne 'safety_regression_report_v1') {
            $violations += "Unexpected safety regression report schema: $($safetyReport.schema)"
        }
    }
    catch {
        $violations += "Failed to parse safety regression report: $($_.Exception.Message)"
    }
}

if ($breakglassActiveEntries -gt 0) {
    Write-Warning "BreakGlass override is active. activeEntries=$breakglassActiveEntries"
    if ($declaredClass -ne 'Class-D') {
        $violations += "BreakGlass override is active but declared class is not Class-D: declaredClass=$declaredClass activeEntries=$breakglassActiveEntries"
    }
    if ($Tier -ne 'exhaustive') {
        $violations += "BreakGlass override requires exhaustive tier: tier=$Tier"
    }
    if ($null -eq $safetyReport) {
        $violations += "BreakGlass override requires safety regression evidence: missing $safetyReportPath"
    }
    elseif (-not [bool]$safetyReport.safetyPass) {
        $violations += 'BreakGlass override requires SafetyPass=true'
    }
}

if ($declaredClass -eq 'Class-D') {
    if ($null -eq $breakglassReport) {
        $violations += "Class-D requires breakglass evidence report: missing $breakglassReportPath"
    }
    elseif ($breakglassActiveEntries -lt 1) {
        $violations += 'Class-D requires at least one active BreakGlass override entry'
    }

    if ($null -eq $safetyReport) {
        $violations += "Class-D requires safety regression evidence report: missing $safetyReportPath"
    }
    elseif (-not [bool]$safetyReport.safetyPass) {
        $violations += 'Class-D requires SafetyPass=true'
    }
}

if ($null -ne $declaredPolicy -and [bool]$declaredPolicy.runtimeCodeChangeZeroRequired -and -not $runtimeCodeChangeZero) {
    $violations += "Policy boolean check failed: runtimeCodeChangeZeroRequired=true class=$declaredClass confidence=$runtimeCodeChangeZeroConfidence"
}

if ($null -ne $declaredPolicy -and [bool]$declaredPolicy.inventoryDiffStructuralInvariantRequired -and -not $inventoryStructuralInvariant) {
    $violations += "Policy boolean check failed: inventoryDiffStructuralInvariantRequired=true class=$declaredClass"
}

$diagnostics = New-Object System.Collections.Generic.List[object]
foreach ($requiredNote in $requiredNotes) {
    $noteSatisfied = $false
    $evidence = [ordered]@{}

    switch ($requiredNote) {
        'runtime code change zero' {
            $noteSatisfied = $runtimeCodeChangeZero
            $evidence['inventoryDiffReportPath'] = $inventoryDiffReportPath
            $evidence['inventoryDiffAvailable'] = ($null -ne $inventoryDiffReport)
            $evidence['authorityInventoryReportPath'] = $authorityInventoryReportPath
            $evidence['authorityInventoryAvailable'] = ($null -ne $authorityInventoryReport)
            $evidence['summary'] = if ($null -ne $inventoryDiffReport) { $inventoryDiffReport.summary } else { $null }
            $evidence['runtimeCodeChangeZeroConfidence'] = $runtimeCodeChangeZeroConfidence
            $evidence['runtimeCodeChangeDetectionReady'] = [bool]$runtimeCodeChangeSignal.detectionReady
            $evidence['runtimeChangedFiles'] = @($runtimeCodeChangeSignal.runtimeChangedFiles)
            $evidence['totalChangedFiles'] = [int]$runtimeCodeChangeSignal.totalChangedFiles
            $evidence['detectionMode'] = "$($runtimeCodeChangeSignal.detectionMode)"
            $evidence['detectionDetail'] = "$($runtimeCodeChangeSignal.detail)"
            if (-not $noteSatisfied) {
                $violations += 'Required note check failed: note=runtime code change zero runtime change detected or fallback invariant not satisfied'
            }
        }
        'inventory diff structural invariant' {
            $noteSatisfied = $inventoryStructuralInvariant
            $evidence['inventoryDiffReportPath'] = $inventoryDiffReportPath
            $evidence['inventoryDiffAvailable'] = ($null -ne $inventoryDiffReport)
            $evidence['summary'] = if ($null -ne $inventoryDiffReport) { $inventoryDiffReport.summary } else { $null }
            if (-not $noteSatisfied) {
                $violations += 'Required note check failed: note=inventory diff structural invariant inventory diff summary is not structurally invariant'
            }
        }
        'soak short 30m' {
            $noteSatisfied = ($SoakMinutes -ge 30)
            $evidence['requiredMinutes'] = 30
            $evidence['actualMinutes'] = $SoakMinutes
            if (-not $noteSatisfied) {
                $violations += "Required note check failed: note=soak short 30m requiredMinutes=30 actualMinutes=$SoakMinutes"
            }
        }
        'soak long 4h' {
            $noteSatisfied = ($SoakMinutes -ge 240)
            $evidence['requiredMinutes'] = 240
            $evidence['actualMinutes'] = $SoakMinutes
            if (-not $noteSatisfied) {
                $violations += "Required note check failed: note=soak long 4h requiredMinutes=240 actualMinutes=$SoakMinutes"
            }
        }
        'soak medium 24h' {
            $noteSatisfied = ($SoakMinutes -ge 1440)
            $evidence['requiredMinutes'] = 1440
            $evidence['actualMinutes'] = $SoakMinutes
            if (-not $noteSatisfied) {
                $violations += "Required note check failed: note=soak medium 24h requiredMinutes=1440 actualMinutes=$SoakMinutes"
            }
        }
        'soak long 72h' {
            $noteSatisfied = ($SoakMinutes -ge 4320)
            $evidence['requiredMinutes'] = 4320
            $evidence['actualMinutes'] = $SoakMinutes
            if (-not $noteSatisfied) {
                $violations += "Required note check failed: note=soak long 72h requiredMinutes=4320 actualMinutes=$SoakMinutes"
            }
        }
        'soak extreme 1week' {
            $noteSatisfied = ($SoakMinutes -ge 10080)
            $evidence['requiredMinutes'] = 10080
            $evidence['actualMinutes'] = $SoakMinutes
            if (-not $noteSatisfied) {
                $violations += "Required note check failed: note=soak extreme 1week requiredMinutes=10080 actualMinutes=$SoakMinutes"
            }
        }
        'break-glass approval' {
            $noteSatisfied = ($breakglassActiveEntries -gt 0) -and $breakglassReportReady
            $evidence['breakglassReportPath'] = $breakglassReportPath
            $evidence['breakglassActiveEntries'] = $breakglassActiveEntries
            $evidence['breakglassReportReady'] = $breakglassReportReady
            if (-not $noteSatisfied) {
                $violations += 'Required note check failed: note=break-glass approval breakglass report is not ready with active entries'
            }
        }
        'rollback plan required' {
            $noteSatisfied = ($breakglassActiveEntries -gt 0) -and $breakglassReportReady
            $evidence['breakglassReportPath'] = $breakglassReportPath
            $evidence['breakglassActiveEntries'] = $breakglassActiveEntries
            $evidence['breakglassReportReady'] = $breakglassReportReady
            if (-not $noteSatisfied) {
                $violations += 'Required note check failed: note=rollback plan required breakglass report is not ready with active entries'
            }
        }
        default {
            $evidence['requiredNote'] = $requiredNote
            $violations += "Unknown required note in PR SLA policy: class=$declaredClass note=$requiredNote"
        }
    }

    $diagnostics.Add(@{
            note = $requiredNote
            satisfied = $noteSatisfied
            evidence = $evidence
        }) | Out-Null
}

$openedAtValue = $null
$openedAtSource = 'none'
if (-not [string]::IsNullOrWhiteSpace($OpenedAt)) {
    $openedAtValue = [datetime]::Parse($OpenedAt)
    $openedAtSource = 'explicit-argument'
}
else {
    $openedAtFromEvent = Resolve-OpenedAtFromEvent -EventPath $env:GITHUB_EVENT_PATH
    if ($null -ne $openedAtFromEvent) {
        $openedAtValue = $openedAtFromEvent
        $openedAtSource = 'github-event'
    }
}

$nowValue = if ([string]::IsNullOrWhiteSpace($Now)) { Get-Date } else { [datetime]::Parse($Now) }

$deadlineAt = $null
$needsRevalidation = $false
$labelSuggestions = @()
$eventHeadSha = Resolve-EventHeadSha -EventPath $env:GITHUB_EVENT_PATH
$currentHeadSha = if (-not [string]::IsNullOrWhiteSpace($env:GITHUB_SHA)) {
    $env:GITHUB_SHA
}
else {
    try {
        "$(& git -C $repoRoot rev-parse HEAD)".Trim()
    }
    catch {
        $null
    }
}
$staleEvaluation = $false

if (-not [string]::IsNullOrWhiteSpace($eventHeadSha) -and -not [string]::IsNullOrWhiteSpace($currentHeadSha) -and $eventHeadSha -ne $currentHeadSha) {
    $staleEvaluation = $true
    $needsRevalidation = $true
    if ($labelSuggestions -notcontains 'needs-revalidation') {
        $labelSuggestions += 'needs-revalidation'
    }
    $violations += "PR SLA evaluation is stale: eventHeadSha=$eventHeadSha currentHeadSha=$currentHeadSha"
}

if ($null -ne $declaredPolicy -and $null -ne $openedAtValue) {
    $deadlineAt = Add-BusinessDays -Start $openedAtValue -BusinessDays ([int]$declaredPolicy.deadlineBusinessDays)
    if ($nowValue -gt $deadlineAt) {
        $needsRevalidation = $true
        $labelSuggestions += 'needs-revalidation'
        $violations += "SLA deadline exceeded for ${declaredClass}: openedAt=$($openedAtValue.ToString('o')) deadlineAt=$($deadlineAt.ToString('o')) now=$($nowValue.ToString('o'))"
    }
}

$effectiveGitRefValue = if ([string]::IsNullOrWhiteSpace($effectiveGitRef)) { $null } else { $effectiveGitRef }
$safetyPassValue = if ($null -ne $safetyReport) { [bool]$safetyReport.safetyPass } else { $null }
$openedAtIsoValue = if ($null -ne $openedAtValue) { $openedAtValue.ToString('o') } else { $null }
$deadlineAtIsoValue = if ($null -ne $deadlineAt) { $deadlineAt.ToString('o') } else { $null }

$diagnosticsArray = @()
foreach ($item in $diagnostics) {
    $diagnosticsArray += [pscustomobject]$item
}

$report = [pscustomobject]@{}
$report | Add-Member -NotePropertyName 'schema' -NotePropertyValue 'pr_sla_report_v1'
$report | Add-Member -NotePropertyName 'generatedAt' -NotePropertyValue (Get-Date -Format 'o')
$report | Add-Member -NotePropertyName 'declaredClass' -NotePropertyValue $declaredClass
$report | Add-Member -NotePropertyName 'declaredClassSource' -NotePropertyValue $declaredClassSource
$report | Add-Member -NotePropertyName 'tier' -NotePropertyValue $Tier
$report | Add-Member -NotePropertyName 'policyPath' -NotePropertyValue $policyFullPath
$report | Add-Member -NotePropertyName 'policy' -NotePropertyValue $policy
$report | Add-Member -NotePropertyName 'releaseRequiresExhaustive' -NotePropertyValue ([bool]$policy.releaseRequiresExhaustive)
$report | Add-Member -NotePropertyName 'releaseWindow' -NotePropertyValue $releaseWindowActive
$report | Add-Member -NotePropertyName 'gitRef' -NotePropertyValue $effectiveGitRefValue
$report | Add-Member -NotePropertyName 'breakglassReportPath' -NotePropertyValue $breakglassReportPath
$report | Add-Member -NotePropertyName 'breakglassActiveEntries' -NotePropertyValue $breakglassActiveEntries
$report | Add-Member -NotePropertyName 'breakglassReportReady' -NotePropertyValue $breakglassReportReady
$report | Add-Member -NotePropertyName 'safetyRegressionReportPath' -NotePropertyValue $safetyReportPath
$report | Add-Member -NotePropertyName 'safetyPass' -NotePropertyValue $safetyPassValue
$report | Add-Member -NotePropertyName 'authorityInventoryReportPath' -NotePropertyValue $authorityInventoryReportPath
$report | Add-Member -NotePropertyName 'inventoryDiffReportPath' -NotePropertyValue $inventoryDiffReportPath
$report | Add-Member -NotePropertyName 'soakMinutes' -NotePropertyValue $SoakMinutes
$report | Add-Member -NotePropertyName 'runtimeCodeChangeZero' -NotePropertyValue $runtimeCodeChangeZero
$report | Add-Member -NotePropertyName 'runtimeCodeChangeZeroConfidence' -NotePropertyValue $runtimeCodeChangeZeroConfidence
$report | Add-Member -NotePropertyName 'runtimeCodeChangeSignal' -NotePropertyValue $runtimeCodeChangeSignal
$report | Add-Member -NotePropertyName 'requiredNotes' -NotePropertyValue @($requiredNotes)
$report | Add-Member -NotePropertyName 'openedAt' -NotePropertyValue $openedAtIsoValue
$report | Add-Member -NotePropertyName 'openedAtSource' -NotePropertyValue $openedAtSource
$report | Add-Member -NotePropertyName 'eventHeadSha' -NotePropertyValue $eventHeadSha
$report | Add-Member -NotePropertyName 'currentHeadSha' -NotePropertyValue $currentHeadSha
$report | Add-Member -NotePropertyName 'staleEvaluation' -NotePropertyValue $staleEvaluation
$report | Add-Member -NotePropertyName 'now' -NotePropertyValue ($nowValue.ToString('o'))
$report | Add-Member -NotePropertyName 'deadlineAt' -NotePropertyValue $deadlineAtIsoValue
$report | Add-Member -NotePropertyName 'needsRevalidation' -NotePropertyValue $needsRevalidation
$report | Add-Member -NotePropertyName 'labelSuggestions' -NotePropertyValue @($labelSuggestions)
$report | Add-Member -NotePropertyName 'diagnostics' -NotePropertyValue $diagnosticsArray
$report | Add-Member -NotePropertyName 'violations' -NotePropertyValue @($violations)
$report | Add-Member -NotePropertyName 'ready' -NotePropertyValue ($violations.Count -eq 0)

$report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $evidencePath -Encoding UTF8
Write-Host "[INFO] PR SLA report written: $evidencePath"

if ($violations.Count -gt 0) {
    foreach ($violation in $violations) {
        Write-Host "[ERROR] $violation"
    }
    throw "PR SLA verification failed. violations=$($violations.Count)"
}

Write-Host '[PASS] PR SLA verification passed'
