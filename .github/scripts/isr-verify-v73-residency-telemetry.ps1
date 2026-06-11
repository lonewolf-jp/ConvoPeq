param(
    [ValidateSet('warn', 'fail')]
    [string]$Mode = ''
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$policyPath = Join-Path $repoRoot '.github\isr-ai-governance-policy.json'
$evidenceDir = Join-Path $repoRoot 'evidence'
$reportPath = Join-Path $evidenceDir 'isr_v73_residency_telemetry_report.json'

$policy = Get-Content -LiteralPath $policyPath -Raw -Encoding UTF8 | ConvertFrom-Json
if ("$($policy.schema)" -ne 'isr_ai_governance_policy_v1') {
    throw "Unexpected policy schema: $($policy.schema)"
}

function ConvertTo-IsoDateStrict {
    param(
        [Parameter(Mandatory = $true)][string]$Raw,
        [Parameter(Mandatory = $true)][string]$Context
    )

    try {
        return [datetime]::ParseExact($Raw, 'yyyy-MM-dd', [System.Globalization.CultureInfo]::InvariantCulture)
    }
    catch {
        throw "Invalid date format at ${Context}: '$Raw' (expected yyyy-MM-dd)"
    }
}

function Assert-AllowlistContract {
    param(
        [Parameter(Mandatory = $true)]$Entries,
        [Parameter(Mandatory = $true)][string]$Context,
        [Parameter(Mandatory = $true)][string[]]$RequiredFields
    )

    $today = (Get-Date).Date
    $index = 0
    foreach ($entry in @($Entries)) {
        $index++
        foreach ($field in $RequiredFields) {
            $value = "$($entry.$field)"
            if ([string]::IsNullOrWhiteSpace($value)) {
                throw "$Context[$index] missing required field: $field"
            }
        }

        $expiry = ConvertTo-IsoDateStrict -Raw "$($entry.expiry)" -Context "$Context[$index].expiry"
        if ($today -gt $expiry.Date) {
            throw "$Context[$index] expired: expiry=$($entry.expiry)"
        }
    }
}

function Assert-RegexCompiles {
    param(
        [Parameter(Mandatory = $true)][string]$Pattern,
        [Parameter(Mandatory = $true)][string]$Context
    )

    if ([string]::IsNullOrWhiteSpace($Pattern)) {
        return
    }

    try {
        [void][System.Text.RegularExpressions.Regex]::new($Pattern)
    }
    catch {
        throw "Invalid regex at ${Context}: '$Pattern'"
    }
}

function Assert-RegexFields {
    param(
        [Parameter(Mandatory = $true)]$Entries,
        [Parameter(Mandatory = $true)][string]$Context,
        [Parameter(Mandatory = $true)][string[]]$Fields
    )

    $index = 0
    foreach ($entry in @($Entries)) {
        $index++
        foreach ($field in $Fields) {
            $value = "$($entry.$field)"
            Assert-RegexCompiles -Pattern $value -Context "$Context[$index].$field"
        }
    }
}

foreach ($field in @('owner', 'issue', 'rationale', 'expiry')) {
    if ($null -eq $policy.$field -or [string]::IsNullOrWhiteSpace("$($policy.$field)")) {
        throw "Policy missing required field: $field"
    }
}

$policyExpiry = ConvertTo-IsoDateStrict -Raw "$($policy.expiry)" -Context 'policy.expiry'
if ((Get-Date).Date -gt $policyExpiry.Date) {
    throw "Policy expired: expiry=$($policy.expiry)"
}

Assert-AllowlistContract -Entries @($policy.residencyContainerAllowlist) -Context 'residencyContainerAllowlist' -RequiredFields @('pathRegex', 'lineRegex', 'owner', 'issue', 'rationale', 'expiry')
Assert-AllowlistContract -Entries @($policy.telemetryCounterOwners) -Context 'telemetryCounterOwners' -RequiredFields @('counterRegex', 'allowedPathRegex', 'owner', 'incrementCondition', 'decrementOrResetCondition', 'shutdownFinalizationBehavior', 'issue', 'rationale', 'expiry')
Assert-AllowlistContract -Entries @($policy.residencyOwnershipTable) -Context 'residencyOwnershipTable' -RequiredFields @('name', 'producer', 'owner', 'reclaimTrigger', 'drainAuthority', 'shutdownCompletionPath', 'issue', 'rationale', 'expiry')
Assert-AllowlistContract -Entries @($policy.residencyTelemetryChecks.crossfadeRtForbiddenWriteScopes) -Context 'residencyTelemetryChecks.crossfadeRtForbiddenWriteScopes' -RequiredFields @('pathRegex', 'owner', 'issue', 'rationale', 'expiry')
Assert-AllowlistContract -Entries @($policy.residencyTelemetryChecks.requiredCrossfadeAuthorityApplications) -Context 'residencyTelemetryChecks.requiredCrossfadeAuthorityApplications' -RequiredFields @('pathRegex', 'lineRegex', 'owner', 'issue', 'rationale', 'expiry')
Assert-AllowlistContract -Entries @($policy.residencyTelemetryChecks.crossfadeAuthorityAllowedWriteScopes) -Context 'residencyTelemetryChecks.crossfadeAuthorityAllowedWriteScopes' -RequiredFields @('pathRegex', 'owner', 'issue', 'rationale', 'expiry')
Assert-AllowlistContract -Entries @($policy.residencyTelemetryChecks.crossfadeAtomicReadAllowlist) -Context 'residencyTelemetryChecks.crossfadeAtomicReadAllowlist' -RequiredFields @('pathRegex', 'lineRegex', 'owner', 'issue', 'rationale', 'expiry')
Assert-AllowlistContract -Entries @($policy.residencyTelemetryChecks.requiredCrossfadePreparedSnapshotApplications) -Context 'residencyTelemetryChecks.requiredCrossfadePreparedSnapshotApplications' -RequiredFields @('pathRegex', 'lineRegex', 'owner', 'issue', 'rationale', 'expiry')
Assert-AllowlistContract -Entries @($policy.residencyTelemetryChecks.requiredCrossfadePreparedSnapshotConsumers) -Context 'residencyTelemetryChecks.requiredCrossfadePreparedSnapshotConsumers' -RequiredFields @('pathRegex', 'lineRegex', 'owner', 'issue', 'rationale', 'expiry')
Assert-AllowlistContract -Entries @($policy.residencyTelemetryChecks.crossfadePreparedSnapshotConsumerAllowedScopes) -Context 'residencyTelemetryChecks.crossfadePreparedSnapshotConsumerAllowedScopes' -RequiredFields @('pathRegex', 'owner', 'issue', 'rationale', 'expiry')
Assert-AllowlistContract -Entries @($policy.residencyTelemetryChecks.crossfadePreparedStateFieldUsageAllowedScopes) -Context 'residencyTelemetryChecks.crossfadePreparedStateFieldUsageAllowedScopes' -RequiredFields @('pathRegex', 'owner', 'issue', 'rationale', 'expiry')
Assert-AllowlistContract -Entries @($policy.residencyTelemetryChecks.requiredCrossfadePreparedStateFieldApplications) -Context 'residencyTelemetryChecks.requiredCrossfadePreparedStateFieldApplications' -RequiredFields @('pathRegex', 'lineRegex', 'owner', 'issue', 'rationale', 'expiry')
Assert-AllowlistContract -Entries @($policy.residencyTelemetryChecks.forbiddenTelemetryAuthorityScopes) -Context 'residencyTelemetryChecks.forbiddenTelemetryAuthorityScopes' -RequiredFields @('pathRegex', 'owner', 'issue', 'rationale', 'expiry')
Assert-AllowlistContract -Entries @($policy.residencyTelemetryChecks.requiredTelemetryApplications) -Context 'residencyTelemetryChecks.requiredTelemetryApplications' -RequiredFields @('pathRegex', 'lineRegex', 'owner', 'issue', 'rationale', 'expiry')
Assert-RegexFields -Entries @($policy.residencyContainerAllowlist) -Context 'residencyContainerAllowlist' -Fields @('pathRegex', 'lineRegex')
Assert-RegexFields -Entries @($policy.telemetryCounterOwners) -Context 'telemetryCounterOwners' -Fields @('counterRegex', 'allowedPathRegex')
Assert-RegexFields -Entries @($policy.residencyTelemetryChecks.crossfadeRtForbiddenWriteScopes) -Context 'residencyTelemetryChecks.crossfadeRtForbiddenWriteScopes' -Fields @('pathRegex')
Assert-RegexFields -Entries @($policy.residencyTelemetryChecks.requiredCrossfadeAuthorityApplications) -Context 'residencyTelemetryChecks.requiredCrossfadeAuthorityApplications' -Fields @('pathRegex', 'lineRegex')
Assert-RegexFields -Entries @($policy.residencyTelemetryChecks.crossfadeAuthorityAllowedWriteScopes) -Context 'residencyTelemetryChecks.crossfadeAuthorityAllowedWriteScopes' -Fields @('pathRegex')
Assert-RegexFields -Entries @($policy.residencyTelemetryChecks.crossfadeAtomicReadAllowlist) -Context 'residencyTelemetryChecks.crossfadeAtomicReadAllowlist' -Fields @('pathRegex', 'lineRegex')
Assert-RegexFields -Entries @($policy.residencyTelemetryChecks.requiredCrossfadePreparedSnapshotApplications) -Context 'residencyTelemetryChecks.requiredCrossfadePreparedSnapshotApplications' -Fields @('pathRegex', 'lineRegex')
Assert-RegexFields -Entries @($policy.residencyTelemetryChecks.requiredCrossfadePreparedSnapshotConsumers) -Context 'residencyTelemetryChecks.requiredCrossfadePreparedSnapshotConsumers' -Fields @('pathRegex', 'lineRegex')
Assert-RegexFields -Entries @($policy.residencyTelemetryChecks.crossfadePreparedSnapshotConsumerAllowedScopes) -Context 'residencyTelemetryChecks.crossfadePreparedSnapshotConsumerAllowedScopes' -Fields @('pathRegex')
Assert-RegexFields -Entries @($policy.residencyTelemetryChecks.crossfadePreparedStateFieldUsageAllowedScopes) -Context 'residencyTelemetryChecks.crossfadePreparedStateFieldUsageAllowedScopes' -Fields @('pathRegex')
Assert-RegexFields -Entries @($policy.residencyTelemetryChecks.requiredCrossfadePreparedStateFieldApplications) -Context 'residencyTelemetryChecks.requiredCrossfadePreparedStateFieldApplications' -Fields @('pathRegex', 'lineRegex')
Assert-RegexFields -Entries @($policy.residencyTelemetryChecks.forbiddenTelemetryAuthorityScopes) -Context 'residencyTelemetryChecks.forbiddenTelemetryAuthorityScopes' -Fields @('pathRegex')
Assert-RegexFields -Entries @($policy.residencyTelemetryChecks.requiredTelemetryApplications) -Context 'residencyTelemetryChecks.requiredTelemetryApplications' -Fields @('pathRegex', 'lineRegex')

foreach ($pattern in @($policy.residencyTelemetryChecks.requiredTelemetryPatterns)) {
    Assert-RegexCompiles -Pattern "$pattern" -Context 'residencyTelemetryChecks.requiredTelemetryPatterns[]'
}
foreach ($pattern in @($policy.residencyTelemetryChecks.crossfadeRtForbiddenWritePatterns)) {
    Assert-RegexCompiles -Pattern "$pattern" -Context 'residencyTelemetryChecks.crossfadeRtForbiddenWritePatterns[]'
}
foreach ($pattern in @($policy.residencyTelemetryChecks.forbiddenTelemetryAuthorityPatterns)) {
    Assert-RegexCompiles -Pattern "$pattern" -Context 'residencyTelemetryChecks.forbiddenTelemetryAuthorityPatterns[]'
}

foreach ($contract in @($policy.residencyBoundednessContracts)) {
    Assert-RegexCompiles -Pattern "$($contract.hardUpperBoundPattern)" -Context "residencyBoundednessContracts[$($contract.name)].hardUpperBoundPattern"
    Assert-RegexCompiles -Pattern "$($contract.warnThresholdPattern)" -Context "residencyBoundednessContracts[$($contract.name)].warnThresholdPattern"
    Assert-RegexCompiles -Pattern "$($contract.forceDrainTriggerPattern)" -Context "residencyBoundednessContracts[$($contract.name)].forceDrainTriggerPattern"
}

if (-not (Test-Path -LiteralPath $evidenceDir)) {
    New-Item -Path $evidenceDir -ItemType Directory | Out-Null
}

function Get-RelativePathCompat {
    param([string]$BasePath, [string]$TargetPath)
    $baseFull = [System.IO.Path]::GetFullPath($BasePath)
    $targetFull = [System.IO.Path]::GetFullPath($TargetPath)
    if ($targetFull.StartsWith($baseFull, [System.StringComparison]::OrdinalIgnoreCase)) {
        return $targetFull.Substring($baseFull.Length).TrimStart([char[]]@([char]'\', [char]'/')).Replace('\', '/')
    }
    return $targetFull.Replace('\', '/')
}

$effectiveMode = $Mode
if ([string]::IsNullOrWhiteSpace($effectiveMode)) {
    $effectiveMode = "$($policy.residencyTelemetryChecks.phase)"
    if ([string]::IsNullOrWhiteSpace($effectiveMode)) { $effectiveMode = 'warn' }
}

if (@('warn', 'fail') -notcontains $effectiveMode) {
    throw "Invalid residencyTelemetryChecks.phase: $effectiveMode (expected warn|fail)"
}

$sourceRoot = Join-Path $repoRoot 'src'
$allSourceFiles = Get-ChildItem -Path $sourceRoot -Recurse -File
$audioEngineHeaderPath = Join-Path $repoRoot 'src\audioengine\AudioEngine.h'
$sourceRelativePaths = @()
foreach ($f in @($allSourceFiles)) {
    $sourceRelativePaths += Get-RelativePathCompat -BasePath $repoRoot -TargetPath $f.FullName
}

function Test-RegexMatchesAnyPath {
    param(
        [Parameter(Mandatory = $true)][string]$PathRegex,
        [Parameter(Mandatory = $true)][string[]]$RelativePaths
    )

    foreach ($rp in $RelativePaths) {
        if ([System.Text.RegularExpressions.Regex]::IsMatch($rp, $PathRegex)) {
            return $true
        }
    }

    return $false
}

$residencyDeclarationFiles = @()
if (Test-Path -LiteralPath $sourceRoot) {
    $residencyDeclarationFiles += Get-ChildItem -Path $sourceRoot -Recurse -File -Include '*.h', '*.hpp'
}

$residencyViolations = @()
$residencyOwnershipViolations = @()
$residencyAuthorityViolations = @()

$knownContainerSet = New-Object 'System.Collections.Generic.HashSet[string]' ([System.StringComparer]::OrdinalIgnoreCase)
foreach ($name in @($policy.residencyKnownContainers)) {
    if (-not [string]::IsNullOrWhiteSpace("$name")) {
        [void]$knownContainerSet.Add("$name")
    }
}

$boundedContainerSet = New-Object 'System.Collections.Generic.HashSet[string]' ([System.StringComparer]::OrdinalIgnoreCase)
foreach ($contract in @($policy.residencyBoundednessContracts)) {
    $name = "$($contract.name)"
    if (-not [string]::IsNullOrWhiteSpace($name)) {
        [void]$boundedContainerSet.Add($name)
    }
}

$ownershipContainerSet = New-Object 'System.Collections.Generic.HashSet[string]' ([System.StringComparer]::OrdinalIgnoreCase)
foreach ($entry in @($policy.residencyOwnershipTable)) {
    $name = "$($entry.name)"
    if (-not [string]::IsNullOrWhiteSpace($name)) {
        [void]$ownershipContainerSet.Add($name)
    }
}

foreach ($known in $knownContainerSet) {
    if (-not $ownershipContainerSet.Contains($known)) {
        $residencyOwnershipViolations += [ordered]@{
            file = 'policy'
            line = 0
            snippet = $known
            checkId = 'CI-RESIDENCY-002'
            message = "known residency container missing ownership contract: $known"
        }
    }

    if (-not $boundedContainerSet.Contains($known)) {
        $residencyOwnershipViolations += [ordered]@{
            file = 'policy'
            line = 0
            snippet = $known
            checkId = 'CI-RESIDENCY-002'
            message = "known residency container missing boundedness contract: $known"
        }
    }
}

foreach ($owned in $ownershipContainerSet) {
    if (-not $knownContainerSet.Contains($owned)) {
        $residencyOwnershipViolations += [ordered]@{
            file = 'policy'
            line = 0
            snippet = $owned
            checkId = 'CI-RESIDENCY-002'
            message = "ownership contract targets unknown residency container: $owned"
        }
    }
}

$hasResidencyAuthorityEnum = $false
foreach ($declFile in $residencyDeclarationFiles) {
    if (Select-String -Path $declFile.FullName -Pattern 'enum\s+class\s+ResidencyAuthority\b' -Quiet -Encoding UTF8) {
        $hasResidencyAuthorityEnum = $true
        break
    }
}

if (-not $hasResidencyAuthorityEnum) {
    $residencyAuthorityViolations += [ordered]@{
        file = 'src/**'
        line = 0
        snippet = 'enum class ResidencyAuthority'
        checkId = 'CI-RESIDENCY-003'
        message = 'ResidencyAuthority enum contract is missing'
    }
}
else {
    $requiredResidencyAuthorityValues = @('PublicationCoordinator', 'DeferredDeleteFallback', 'EpochRetire', 'ShutdownDrain')
    foreach ($value in $requiredResidencyAuthorityValues) {
        $valueMatched = $false
        foreach ($declFile in $residencyDeclarationFiles) {
            if (Select-String -Path $declFile.FullName -Pattern ("\b" + [System.Text.RegularExpressions.Regex]::Escape($value) + "\b") -Quiet -Encoding UTF8) {
                $valueMatched = $true
                break
            }
        }

        if (-not $valueMatched) {
            $residencyAuthorityViolations += [ordered]@{
                file = 'src/**'
                line = 0
                snippet = $value
                checkId = 'CI-RESIDENCY-003'
                message = "ResidencyAuthority enum value is missing: $value"
            }
        }
    }
}

foreach ($contract in @($policy.residencyBoundednessContracts)) {
    $name = "$($contract.name)"
    $hard = "$($contract.hardUpperBoundPattern)"
    $warn = "$($contract.warnThresholdPattern)"
    $drain = "$($contract.forceDrainTriggerPattern)"

    if ([string]::IsNullOrWhiteSpace($name)) { continue }

    foreach ($pair in @(
        @{ Id = 'hardUpperBoundPattern'; Pattern = $hard },
        @{ Id = 'warnThresholdPattern'; Pattern = $warn },
        @{ Id = 'forceDrainTriggerPattern'; Pattern = $drain }
    )) {
        if ([string]::IsNullOrWhiteSpace("$($pair.Pattern)")) {
            $residencyViolations += [ordered]@{
                file = 'policy'
                line = 0
                snippet = $name
                checkId = 'CI-RESIDENCY-001'
                message = "missing policy pattern: $($pair.Id) for residency '$name'"
            }
            continue
        }

        $matched = $false
        foreach ($f in $allSourceFiles) {
            if (Select-String -Path $f.FullName -Pattern "$($pair.Pattern)" -Quiet -Encoding UTF8) {
                $matched = $true
                break
            }
        }

        if (-not $matched) {
            $residencyViolations += [ordered]@{
                file = 'src/**'
                line = 0
                snippet = $name
                checkId = 'CI-RESIDENCY-001'
                message = "pattern not found: $($pair.Id) => $($pair.Pattern)"
            }
        }
    }
}

$residencyDeclRegex = [System.Text.RegularExpressions.Regex]::new('^\s*(?<type>std::vector|std::deque|std::list|LockFree[a-zA-Z0-9_]*Queue|LockFreeRingBuffer)\s*<[^;]+>\s+(?<name>[A-Za-z_][A-Za-z0-9_]*)\s*;')
$residencyNameRegex = [System.Text.RegularExpressions.Regex]::new('(retire|retired|fallback|deferred|queue|queued|staging|epoch)', [System.Text.RegularExpressions.RegexOptions]::IgnoreCase)

foreach ($declFile in $residencyDeclarationFiles) {
    $relativePath = Get-RelativePathCompat -BasePath $repoRoot -TargetPath $declFile.FullName
    $lineNumber = 0
    foreach ($line in (Get-Content -LiteralPath $declFile.FullName -Encoding UTF8)) {
        $lineNumber++
        $declMatch = $residencyDeclRegex.Match("$line")
        if (-not $declMatch.Success) {
            continue
        }

        if (-not $residencyNameRegex.IsMatch("$line")) {
            continue
        }

        $allowedDecl = $false
        foreach ($allow in @($policy.residencyContainerAllowlist)) {
            if ([System.Text.RegularExpressions.Regex]::IsMatch($relativePath, "$($allow.pathRegex)") -and
                [System.Text.RegularExpressions.Regex]::IsMatch("$line", "$($allow.lineRegex)")) {
                $allowedDecl = $true
                break
            }
        }

        if (-not $allowedDecl) {
            $residencyViolations += [ordered]@{
                file = $relativePath
                line = $lineNumber
                snippet = ("$line").Trim()
                checkId = 'CI-RESIDENCY-001'
                message = "residency container declaration is not allowlisted"
            }
        }
    }
}

$telemetryViolations = @()
$telemetryOwnerCoverageViolations = @()
$atomicWritePattern = 'fetchAddAtomic\s*\(|publishAtomic\s*\(|exchangeAtomic\s*\('
foreach ($ownerRule in @($policy.telemetryCounterOwners)) {
    $counterRegex = "$($ownerRule.counterRegex)"
    $allowedPathRegex = "$($ownerRule.allowedPathRegex)"

    if ([string]::IsNullOrWhiteSpace($counterRegex) -or [string]::IsNullOrWhiteSpace($allowedPathRegex)) {
        continue
    }

    $ownerWriteCount = 0

    foreach ($f in $allSourceFiles) {
        $relativePath = Get-RelativePathCompat -BasePath $repoRoot -TargetPath $f.FullName
        $hits = Select-String -Path $f.FullName -Pattern $counterRegex -Encoding UTF8
        foreach ($h in $hits) {
            $lineText = "$($h.Line)"
            if (-not [System.Text.RegularExpressions.Regex]::IsMatch($lineText, $atomicWritePattern)) {
                continue
            }

            $ownerWriteCount++

            if (-not [System.Text.RegularExpressions.Regex]::IsMatch($relativePath, $allowedPathRegex)) {
                $telemetryViolations += [ordered]@{
                    file = $relativePath
                    line = $h.LineNumber
                    snippet = $lineText.Trim()
                    checkId = 'CI-TELEMETRY-001'
                    message = "owner path violation for counterRegex=$counterRegex"
                }
            }
        }
    }

    if ($ownerWriteCount -eq 0) {
        $telemetryOwnerCoverageViolations += [ordered]@{
            file = 'src/**'
            line = 0
            snippet = $counterRegex
            checkId = 'CI-TELEMETRY-003'
            message = "telemetry owner rule has no atomic write coverage: counterRegex=$counterRegex allowedPathRegex=$allowedPathRegex"
        }
    }
}

$requiredTelemetryPatternViolations = @()
foreach ($required in @($policy.residencyTelemetryChecks.requiredTelemetryPatterns)) {
    if ([string]::IsNullOrWhiteSpace("$required")) { continue }

    $matched = $false
    foreach ($f in $allSourceFiles) {
        if (Select-String -Path $f.FullName -Pattern "$required" -Quiet -Encoding UTF8) {
            $matched = $true
            break
        }
    }

    if (-not $matched) {
        $requiredTelemetryPatternViolations += [ordered]@{
            file = 'src/**'
            line = 0
            snippet = ''
            checkId = 'CI-TELEMETRY-002'
            message = "required telemetry pattern not found: $required"
        }
    }
}

$requiredTelemetryApplicationViolations = @()
foreach ($requiredApp in @($policy.residencyTelemetryChecks.requiredTelemetryApplications)) {
    $pathRegex = "$($requiredApp.pathRegex)"
    $lineRegex = "$($requiredApp.lineRegex)"

    if ([string]::IsNullOrWhiteSpace($pathRegex) -or [string]::IsNullOrWhiteSpace($lineRegex)) {
        $requiredTelemetryApplicationViolations += [ordered]@{
            file = 'policy'
            line = 0
            snippet = ''
            checkId = 'CI-TELEMETRY-002'
            message = 'invalid requiredTelemetryApplications entry (pathRegex/lineRegex required)'
        }
        continue
    }

    $matched = $false
    foreach ($f in $allSourceFiles) {
        $relativePath = Get-RelativePathCompat -BasePath $repoRoot -TargetPath $f.FullName
        if (-not [System.Text.RegularExpressions.Regex]::IsMatch($relativePath, $pathRegex)) {
            continue
        }

        if (Select-String -Path $f.FullName -Pattern $lineRegex -Quiet -Encoding UTF8) {
            $matched = $true
            break
        }
    }

    if (-not $matched) {
        $requiredTelemetryApplicationViolations += [ordered]@{
            file = 'src/**'
            line = 0
            snippet = ''
            checkId = 'CI-TELEMETRY-002'
            message = "required telemetry application not found: pathRegex=$pathRegex lineRegex=$lineRegex"
        }
    }
}

$telemetryImmediateSyncViolations = @()
$telemetryImmediateSyncPolicyCoverageViolations = @()
if (-not (Test-Path -LiteralPath $audioEngineHeaderPath)) {
    $telemetryImmediateSyncViolations += [ordered]@{
        file = 'src/audioengine/AudioEngine.h'
        line = 0
        snippet = ''
        checkId = 'CI-TELEMETRY-009'
        message = 'AudioEngine header not found for immediate telemetry sync contract checks'
    }
}
else {
    $audioEngineHeaderText = Get-Content -LiteralPath $audioEngineHeaderPath -Raw -Encoding UTF8

    $requiredImmediateSyncPatterns = @(
        'convo::publishAtomic\(retireQueueDepth_, retireDepth,\s*std::memory_order_release\)',
        'runtimePublicationBridge_\.setRetireBacklogCount\(retireDepth\)'
    )

    foreach ($pattern in $requiredImmediateSyncPatterns) {
        if (-not [System.Text.RegularExpressions.Regex]::IsMatch($audioEngineHeaderText, $pattern)) {
            $telemetryImmediateSyncViolations += [ordered]@{
                file = 'src/audioengine/AudioEngine.h'
                line = 0
                snippet = $pattern
                checkId = 'CI-TELEMETRY-009'
                message = 'missing immediate telemetry sync pattern in fallback enqueue path'
            }
        }

        $policyCovered = $false
        foreach ($requiredApp in @($policy.residencyTelemetryChecks.requiredTelemetryApplications)) {
            $pathRegex = "$($requiredApp.pathRegex)"
            $lineRegex = "$($requiredApp.lineRegex)"
            if (($pathRegex -eq '^src/audioengine/AudioEngine\.h$' -or $pathRegex -eq '^src/audioengine/AudioEngine\.Retire\.cpp$') -and $lineRegex -eq $pattern) {
                $policyCovered = $true
                break
            }
        }

        if (-not $policyCovered) {
            $telemetryImmediateSyncPolicyCoverageViolations += [ordered]@{
                file = 'policy'
                line = 0
                snippet = $pattern
                checkId = 'CI-TELEMETRY-010'
                message = 'immediate telemetry sync contract is not declared in requiredTelemetryApplications policy'
            }
        }
    }
}

$requiredTelemetryOwnerScopeCoverageViolations = @()
foreach ($requiredApp in @($policy.residencyTelemetryChecks.requiredTelemetryApplications)) {
    $requiredPathRegex = "$($requiredApp.pathRegex)"
    if ([string]::IsNullOrWhiteSpace($requiredPathRegex)) {
        continue
    }

    $requiredTargetPaths = @()
    foreach ($f in $allSourceFiles) {
        $relativePath = Get-RelativePathCompat -BasePath $repoRoot -TargetPath $f.FullName
        if ([System.Text.RegularExpressions.Regex]::IsMatch($relativePath, $requiredPathRegex)) {
            $requiredTargetPaths += $relativePath
        }
    }

    if ($requiredTargetPaths.Count -eq 0) {
        continue
    }

    $coveredByOwner = $true
    foreach ($targetPath in $requiredTargetPaths) {
        $pathCovered = $false
        foreach ($ownerRule in @($policy.telemetryCounterOwners)) {
            $allowedPathRegex = "$($ownerRule.allowedPathRegex)"
            if ([string]::IsNullOrWhiteSpace($allowedPathRegex)) {
                continue
            }
            if ([System.Text.RegularExpressions.Regex]::IsMatch($targetPath, $allowedPathRegex)) {
                $pathCovered = $true
                break
            }
        }

        if (-not $pathCovered) {
            $coveredByOwner = $false
            break
        }
    }

    if (-not $coveredByOwner) {
        $requiredTelemetryOwnerScopeCoverageViolations += [ordered]@{
            file = '<policy>'
            line = 0
            snippet = $requiredPathRegex
            checkId = 'CI-TELEMETRY-005'
            message = 'required telemetry application pathRegex is not covered by telemetryCounterOwners.allowedPathRegex'
        }
    }
}

$telemetryOwnerRequiredApplicationCoverageViolations = @()
foreach ($ownerRule in @($policy.telemetryCounterOwners)) {
    $allowedPathRegex = "$($ownerRule.allowedPathRegex)"
    $counterRegex = "$($ownerRule.counterRegex)"
    if ([string]::IsNullOrWhiteSpace($allowedPathRegex)) {
        continue
    }

    $ownerScopePaths = @()
    foreach ($f in $allSourceFiles) {
        $relativePath = Get-RelativePathCompat -BasePath $repoRoot -TargetPath $f.FullName
        if ([System.Text.RegularExpressions.Regex]::IsMatch($relativePath, $allowedPathRegex)) {
            $ownerScopePaths += $relativePath
        }
    }

    if ($ownerScopePaths.Count -eq 0) {
        continue
    }

    $covered = $false
    foreach ($requiredApp in @($policy.residencyTelemetryChecks.requiredTelemetryApplications)) {
        $requiredPathRegex = "$($requiredApp.pathRegex)"
        if ([string]::IsNullOrWhiteSpace($requiredPathRegex)) {
            continue
        }

        foreach ($ownerPath in $ownerScopePaths) {
            if ([System.Text.RegularExpressions.Regex]::IsMatch($ownerPath, $requiredPathRegex)) {
                $covered = $true
                break
            }
        }

        if ($covered) {
            break
        }
    }

    if (-not $covered) {
        $telemetryOwnerRequiredApplicationCoverageViolations += [ordered]@{
            file = '<policy>'
            line = 0
            snippet = "$counterRegex :: $allowedPathRegex"
            checkId = 'CI-TELEMETRY-006'
            message = 'telemetry owner allowed scope has no overlap with required telemetry applications'
        }
    }
}

$residencyContractDuplicateViolations = @()

$residencyOwnershipNameCounts = @{}
foreach ($entry in @($policy.residencyOwnershipTable)) {
    $name = "$($entry.name)"
    if ([string]::IsNullOrWhiteSpace($name)) {
        continue
    }
    if (-not $residencyOwnershipNameCounts.ContainsKey($name)) {
        $residencyOwnershipNameCounts[$name] = 0
    }
    $residencyOwnershipNameCounts[$name]++
}
foreach ($key in $residencyOwnershipNameCounts.Keys) {
    if ($residencyOwnershipNameCounts[$key] -gt 1) {
        $residencyContractDuplicateViolations += [ordered]@{
            file = 'policy'
            line = 0
            snippet = $key
            checkId = 'CI-RESIDENCY-004'
            message = 'duplicate name found in residencyOwnershipTable'
        }
    }
}

$residencyBoundednessNameCounts = @{}
foreach ($entry in @($policy.residencyBoundednessContracts)) {
    $name = "$($entry.name)"
    if ([string]::IsNullOrWhiteSpace($name)) {
        continue
    }
    if (-not $residencyBoundednessNameCounts.ContainsKey($name)) {
        $residencyBoundednessNameCounts[$name] = 0
    }
    $residencyBoundednessNameCounts[$name]++
}
foreach ($key in $residencyBoundednessNameCounts.Keys) {
    if ($residencyBoundednessNameCounts[$key] -gt 1) {
        $residencyContractDuplicateViolations += [ordered]@{
            file = 'policy'
            line = 0
            snippet = $key
            checkId = 'CI-RESIDENCY-004'
            message = 'duplicate name found in residencyBoundednessContracts'
        }
    }
}

$telemetryPolicyDuplicateViolations = @()

$telemetryOwnerKeyCounts = @{}
foreach ($entry in @($policy.telemetryCounterOwners)) {
    $counterRegex = "$($entry.counterRegex)"
    $allowedPathRegex = "$($entry.allowedPathRegex)"
    if ([string]::IsNullOrWhiteSpace($counterRegex) -or [string]::IsNullOrWhiteSpace($allowedPathRegex)) {
        continue
    }
    $key = "$counterRegex || $allowedPathRegex"
    if (-not $telemetryOwnerKeyCounts.ContainsKey($key)) {
        $telemetryOwnerKeyCounts[$key] = 0
    }
    $telemetryOwnerKeyCounts[$key]++
}
foreach ($key in $telemetryOwnerKeyCounts.Keys) {
    if ($telemetryOwnerKeyCounts[$key] -gt 1) {
        $telemetryPolicyDuplicateViolations += [ordered]@{
            file = 'policy'
            line = 0
            snippet = $key
            checkId = 'CI-TELEMETRY-007'
            message = 'duplicate counterRegex+allowedPathRegex found in telemetryCounterOwners'
        }
    }
}

$requiredTelemetryAppKeyCounts = @{}
foreach ($entry in @($policy.residencyTelemetryChecks.requiredTelemetryApplications)) {
    $pathRegex = "$($entry.pathRegex)"
    $lineRegex = "$($entry.lineRegex)"
    if ([string]::IsNullOrWhiteSpace($pathRegex) -or [string]::IsNullOrWhiteSpace($lineRegex)) {
        continue
    }
    $key = "$pathRegex || $lineRegex"
    if (-not $requiredTelemetryAppKeyCounts.ContainsKey($key)) {
        $requiredTelemetryAppKeyCounts[$key] = 0
    }
    $requiredTelemetryAppKeyCounts[$key]++
}
foreach ($key in $requiredTelemetryAppKeyCounts.Keys) {
    if ($requiredTelemetryAppKeyCounts[$key] -gt 1) {
        $telemetryPolicyDuplicateViolations += [ordered]@{
            file = 'policy'
            line = 0
            snippet = $key
            checkId = 'CI-TELEMETRY-008'
            message = 'duplicate pathRegex+lineRegex found in requiredTelemetryApplications'
        }
    }
}

$forbiddenTelemetryAuthorityViolations = @()
$authorityScopeFiles = @()
foreach ($scope in @($policy.residencyTelemetryChecks.forbiddenTelemetryAuthorityScopes)) {
    $pathRegex = "$($scope.pathRegex)"
    if ([string]::IsNullOrWhiteSpace($pathRegex)) {
        continue
    }

    foreach ($f in $allSourceFiles) {
        $relativePath = Get-RelativePathCompat -BasePath $repoRoot -TargetPath $f.FullName
        if ([System.Text.RegularExpressions.Regex]::IsMatch($relativePath, $pathRegex)) {
            $authorityScopeFiles += $f
        }
    }
}

$authorityScopeFiles = @($authorityScopeFiles | Sort-Object FullName -Unique)
foreach ($requiredForbiddenPattern in @($policy.residencyTelemetryChecks.forbiddenTelemetryAuthorityPatterns)) {
    if ([string]::IsNullOrWhiteSpace("$requiredForbiddenPattern")) { continue }

    foreach ($f in $authorityScopeFiles) {
        $relativePath = Get-RelativePathCompat -BasePath $repoRoot -TargetPath $f.FullName
        $hits = Select-String -Path $f.FullName -Pattern "$requiredForbiddenPattern" -Encoding UTF8
        foreach ($h in $hits) {
            $forbiddenTelemetryAuthorityViolations += [ordered]@{
                file = $relativePath
                line = $h.LineNumber
                snippet = "$($h.Line)".Trim()
                checkId = 'CI-TELEMETRY-004'
                message = "forbidden telemetry authority pattern detected: $requiredForbiddenPattern"
            }
        }
    }
}

$crossfadeRtWriteViolations = @()
$crossfadeScopeFiles = @()
foreach ($scope in @($policy.residencyTelemetryChecks.crossfadeRtForbiddenWriteScopes)) {
    $pathRegex = "$($scope.pathRegex)"
    if ([string]::IsNullOrWhiteSpace($pathRegex)) {
        continue
    }

    foreach ($f in $allSourceFiles) {
        $relativePath = Get-RelativePathCompat -BasePath $repoRoot -TargetPath $f.FullName
        if ([System.Text.RegularExpressions.Regex]::IsMatch($relativePath, $pathRegex)) {
            $crossfadeScopeFiles += $f
        }
    }
}

$crossfadeScopeFiles = @($crossfadeScopeFiles | Sort-Object FullName -Unique)
foreach ($forbiddenPattern in @($policy.residencyTelemetryChecks.crossfadeRtForbiddenWritePatterns)) {
    if ([string]::IsNullOrWhiteSpace("$forbiddenPattern")) { continue }

    foreach ($f in $crossfadeScopeFiles) {
        $relativePath = Get-RelativePathCompat -BasePath $repoRoot -TargetPath $f.FullName
        $hits = Select-String -Path $f.FullName -Pattern "$forbiddenPattern" -Encoding UTF8
        foreach ($h in $hits) {
            $crossfadeRtWriteViolations += [ordered]@{
                file = $relativePath
                line = $h.LineNumber
                snippet = "$($h.Line)".Trim()
                checkId = 'CI-CROSSFADE-001'
                message = "forbidden crossfade atomic write detected in RT scope: $forbiddenPattern"
            }
        }
    }
}

$requiredCrossfadeAuthorityApplicationViolations = @()
foreach ($requiredApp in @($policy.residencyTelemetryChecks.requiredCrossfadeAuthorityApplications)) {
    $pathRegex = "$($requiredApp.pathRegex)"
    $lineRegex = "$($requiredApp.lineRegex)"
    if ([string]::IsNullOrWhiteSpace($pathRegex) -or [string]::IsNullOrWhiteSpace($lineRegex)) {
        continue
    }

    $matched = $false
    foreach ($f in $allSourceFiles) {
        $relativePath = Get-RelativePathCompat -BasePath $repoRoot -TargetPath $f.FullName
        if (-not [System.Text.RegularExpressions.Regex]::IsMatch($relativePath, $pathRegex)) {
            continue
        }
        if (Select-String -Path $f.FullName -Pattern $lineRegex -Encoding UTF8) {
            $matched = $true
            break
        }
    }

    if (-not $matched) {
        $requiredCrossfadeAuthorityApplicationViolations += [ordered]@{
            file = '<policy>'
            line = 0
            snippet = "$pathRegex :: $lineRegex"
            checkId = 'CI-CROSSFADE-002'
            message = 'required crossfade authority application not observed in source tree'
        }
    }
}

$crossfadeAuthorityScopeViolations = @()
foreach ($f in $allSourceFiles) {
    $relativePath = Get-RelativePathCompat -BasePath $repoRoot -TargetPath $f.FullName
    $allowedScope = $false
    foreach ($scope in @($policy.residencyTelemetryChecks.crossfadeAuthorityAllowedWriteScopes)) {
        $scopeRegex = "$($scope.pathRegex)"
        if ([string]::IsNullOrWhiteSpace($scopeRegex)) {
            continue
        }
        if ([System.Text.RegularExpressions.Regex]::IsMatch($relativePath, $scopeRegex)) {
            $allowedScope = $true
            break
        }
    }

    if ($allowedScope) {
        continue
    }

    foreach ($forbiddenPattern in @($policy.residencyTelemetryChecks.crossfadeRtForbiddenWritePatterns)) {
        if ([string]::IsNullOrWhiteSpace("$forbiddenPattern")) { continue }
        $hits = Select-String -Path $f.FullName -Pattern "$forbiddenPattern" -Encoding UTF8
        foreach ($h in $hits) {
            $crossfadeAuthorityScopeViolations += [ordered]@{
                file = $relativePath
                line = $h.LineNumber
                snippet = "$($h.Line)".Trim()
                checkId = 'CI-CROSSFADE-003'
                message = "crossfade authority atomic write is outside allowlisted authority scopes"
            }
        }
    }
}

$crossfadeAtomicReadViolations = @()
$crossfadeAtomicReadPattern = 'consumeAtomic\((dspCrossfade(Pending|UseDryAsOld|StartDelayBlocks|DryHoldSamples|DryScaleTarget)|firstIrDryCrossfade(Pending|Done))'
foreach ($f in $allSourceFiles) {
    $relativePath = Get-RelativePathCompat -BasePath $repoRoot -TargetPath $f.FullName
    $hits = Select-String -Path $f.FullName -Pattern $crossfadeAtomicReadPattern -Encoding UTF8
    foreach ($h in $hits) {
        $allowed = $false
        foreach ($rule in @($policy.residencyTelemetryChecks.crossfadeAtomicReadAllowlist)) {
            $pathRegex = "$($rule.pathRegex)"
            $lineRegex = "$($rule.lineRegex)"
            if ([string]::IsNullOrWhiteSpace($pathRegex) -or [string]::IsNullOrWhiteSpace($lineRegex)) {
                continue
            }
            if ([System.Text.RegularExpressions.Regex]::IsMatch($relativePath, $pathRegex) -and
                [System.Text.RegularExpressions.Regex]::IsMatch("$($h.Line)", $lineRegex)) {
                $allowed = $true
                break
            }
        }

        if (-not $allowed) {
            $crossfadeAtomicReadViolations += [ordered]@{
                file = $relativePath
                line = $h.LineNumber
                snippet = "$($h.Line)".Trim()
                checkId = 'CI-CROSSFADE-004'
                message = 'crossfade consumeAtomic read is outside allowlist'
            }
        }
    }
}

$requiredCrossfadePreparedSnapshotViolations = @()
foreach ($requiredApp in @($policy.residencyTelemetryChecks.requiredCrossfadePreparedSnapshotApplications)) {
    $pathRegex = "$($requiredApp.pathRegex)"
    $lineRegex = "$($requiredApp.lineRegex)"
    if ([string]::IsNullOrWhiteSpace($pathRegex) -or [string]::IsNullOrWhiteSpace($lineRegex)) {
        continue
    }

    $matched = $false
    foreach ($f in $allSourceFiles) {
        $relativePath = Get-RelativePathCompat -BasePath $repoRoot -TargetPath $f.FullName
        if (-not [System.Text.RegularExpressions.Regex]::IsMatch($relativePath, $pathRegex)) {
            continue
        }

        if (Select-String -Path $f.FullName -Pattern $lineRegex -Quiet -Encoding UTF8) {
            $matched = $true
            break
        }
    }

    if (-not $matched) {
        $requiredCrossfadePreparedSnapshotViolations += [ordered]@{
            file = '<policy>'
            line = 0
            snippet = "$pathRegex :: $lineRegex"
            checkId = 'CI-CROSSFADE-005'
            message = 'required crossfade prepared-snapshot application not observed in source tree'
        }
    }
}

$requiredCrossfadePreparedSnapshotConsumerViolations = @()
foreach ($requiredApp in @($policy.residencyTelemetryChecks.requiredCrossfadePreparedSnapshotConsumers)) {
    $pathRegex = "$($requiredApp.pathRegex)"
    $lineRegex = "$($requiredApp.lineRegex)"
    if ([string]::IsNullOrWhiteSpace($pathRegex) -or [string]::IsNullOrWhiteSpace($lineRegex)) {
        continue
    }

    $matched = $false
    foreach ($f in $allSourceFiles) {
        $relativePath = Get-RelativePathCompat -BasePath $repoRoot -TargetPath $f.FullName
        if (-not [System.Text.RegularExpressions.Regex]::IsMatch($relativePath, $pathRegex)) {
            continue
        }

        if (Select-String -Path $f.FullName -Pattern $lineRegex -Quiet -Encoding UTF8) {
            $matched = $true
            break
        }
    }

    if (-not $matched) {
        $requiredCrossfadePreparedSnapshotConsumerViolations += [ordered]@{
            file = '<policy>'
            line = 0
            snippet = "$pathRegex :: $lineRegex"
            checkId = 'CI-CROSSFADE-006'
            message = 'required crossfade prepared-snapshot consumer callsite not observed in source tree'
        }
    }
}

$crossfadePreparedConsumerPolicyCoverageViolations = @()
foreach ($requiredConsumer in @($policy.residencyTelemetryChecks.requiredCrossfadePreparedSnapshotConsumers)) {
    $requiredPathRegex = "$($requiredConsumer.pathRegex)"
    if ([string]::IsNullOrWhiteSpace($requiredPathRegex)) {
        continue
    }

    $covered = $false
    foreach ($scope in @($policy.residencyTelemetryChecks.crossfadePreparedSnapshotConsumerAllowedScopes)) {
        $scopeRegex = "$($scope.pathRegex)"
        if ([string]::IsNullOrWhiteSpace($scopeRegex)) {
            continue
        }
        if ($scopeRegex -eq $requiredPathRegex) {
            $covered = $true
            break
        }
    }

    if (-not $covered) {
        $crossfadePreparedConsumerPolicyCoverageViolations += [ordered]@{
            file = '<policy>'
            line = 0
            snippet = $requiredPathRegex
            checkId = 'CI-CROSSFADE-010'
            message = 'required prepared-snapshot consumer pathRegex is not covered by allowed consumer scopes'
        }
    }
}

$crossfadePreparedSnapshotScopeViolations = @()
foreach ($f in $allSourceFiles) {
    $relativePath = Get-RelativePathCompat -BasePath $repoRoot -TargetPath $f.FullName
    if ($relativePath -match '^src/tests/') {
        continue
    }
    $hits = Select-String -Path $f.FullName -Pattern 'consumeCrossfadePreparedSnapshot\(\)' -Encoding UTF8
    foreach ($h in $hits) {
        $allowed = $false
        foreach ($scope in @($policy.residencyTelemetryChecks.crossfadePreparedSnapshotConsumerAllowedScopes)) {
            $scopeRegex = "$($scope.pathRegex)"
            if ([string]::IsNullOrWhiteSpace($scopeRegex)) { continue }
            if ([System.Text.RegularExpressions.Regex]::IsMatch($relativePath, $scopeRegex)) {
                $allowed = $true
                break
            }
        }

        if (-not $allowed) {
            $crossfadePreparedSnapshotScopeViolations += [ordered]@{
                file = $relativePath
                line = $h.LineNumber
                snippet = "$($h.Line)".Trim()
                checkId = 'CI-CROSSFADE-007'
                message = 'consumeCrossfadePreparedSnapshot call is outside allowlisted consumer scopes'
            }
        }
    }
}

$crossfadePreparedStateFieldScopeViolations = @()
foreach ($f in $allSourceFiles) {
    $relativePath = Get-RelativePathCompat -BasePath $repoRoot -TargetPath $f.FullName
    if ($relativePath -match '^src/tests/') {
        continue
    }
    $hits = Select-String -Path $f.FullName -Pattern 'preparedCrossfade\.[A-Za-z_][A-Za-z0-9_]*' -Encoding UTF8
    foreach ($h in $hits) {
        $allowed = $false
        foreach ($scope in @($policy.residencyTelemetryChecks.crossfadePreparedStateFieldUsageAllowedScopes)) {
            $scopeRegex = "$($scope.pathRegex)"
            if ([string]::IsNullOrWhiteSpace($scopeRegex)) { continue }
            if ([System.Text.RegularExpressions.Regex]::IsMatch($relativePath, $scopeRegex)) {
                $allowed = $true
                break
            }
        }

        if (-not $allowed) {
            $crossfadePreparedStateFieldScopeViolations += [ordered]@{
                file = $relativePath
                line = $h.LineNumber
                snippet = "$($h.Line)".Trim()
                checkId = 'CI-CROSSFADE-008'
                message = 'preparedCrossfade field usage is outside allowlisted scopes'
            }
        }
    }
}

$requiredCrossfadePreparedStateFieldApplicationViolations = @()
foreach ($requiredApp in @($policy.residencyTelemetryChecks.requiredCrossfadePreparedStateFieldApplications)) {
    $pathRegex = "$($requiredApp.pathRegex)"
    $lineRegex = "$($requiredApp.lineRegex)"
    if ([string]::IsNullOrWhiteSpace($pathRegex) -or [string]::IsNullOrWhiteSpace($lineRegex)) {
        continue
    }

    $matched = $false
    foreach ($f in $allSourceFiles) {
        $relativePath = Get-RelativePathCompat -BasePath $repoRoot -TargetPath $f.FullName
        if (-not [System.Text.RegularExpressions.Regex]::IsMatch($relativePath, $pathRegex)) {
            continue
        }

        if (Select-String -Path $f.FullName -Pattern $lineRegex -Quiet -Encoding UTF8) {
            $matched = $true
            break
        }
    }

    if (-not $matched) {
        $requiredCrossfadePreparedStateFieldApplicationViolations += [ordered]@{
            file = '<policy>'
            line = 0
            snippet = "$pathRegex :: $lineRegex"
            checkId = 'CI-CROSSFADE-009'
            message = 'required preparedCrossfade field application not observed in source tree'
        }
    }
}

$crossfadePreparedFieldPolicyCoverageViolations = @()
foreach ($requiredFieldApp in @($policy.residencyTelemetryChecks.requiredCrossfadePreparedStateFieldApplications)) {
    $requiredPathRegex = "$($requiredFieldApp.pathRegex)"
    if ([string]::IsNullOrWhiteSpace($requiredPathRegex)) {
        continue
    }

    $covered = $false
    foreach ($scope in @($policy.residencyTelemetryChecks.crossfadePreparedStateFieldUsageAllowedScopes)) {
        $scopeRegex = "$($scope.pathRegex)"
        if ([string]::IsNullOrWhiteSpace($scopeRegex)) {
            continue
        }
        if ($scopeRegex -eq $requiredPathRegex) {
            $covered = $true
            break
        }
    }

    if (-not $covered) {
        $crossfadePreparedFieldPolicyCoverageViolations += [ordered]@{
            file = '<policy>'
            line = 0
            snippet = $requiredPathRegex
            checkId = 'CI-CROSSFADE-011'
            message = 'required preparedCrossfade field pathRegex is not covered by allowed field usage scopes'
        }
    }
}

$crossfadePreparedConsumerToFieldCoverageViolations = @()
foreach ($requiredConsumer in @($policy.residencyTelemetryChecks.requiredCrossfadePreparedSnapshotConsumers)) {
    $consumerPathRegex = "$($requiredConsumer.pathRegex)"
    if ([string]::IsNullOrWhiteSpace($consumerPathRegex)) {
        continue
    }

    $covered = $false
    foreach ($requiredField in @($policy.residencyTelemetryChecks.requiredCrossfadePreparedStateFieldApplications)) {
        $fieldPathRegex = "$($requiredField.pathRegex)"
        if ([string]::IsNullOrWhiteSpace($fieldPathRegex)) {
            continue
        }
        if ($fieldPathRegex -eq $consumerPathRegex) {
            $covered = $true
            break
        }
    }

    if (-not $covered) {
        $crossfadePreparedConsumerToFieldCoverageViolations += [ordered]@{
            file = '<policy>'
            line = 0
            snippet = $consumerPathRegex
            checkId = 'CI-CROSSFADE-012'
            message = 'required prepared-snapshot consumer pathRegex is not covered by required prepared-field applications'
        }
    }
}

$crossfadePreparedFieldToConsumerCoverageViolations = @()
foreach ($requiredField in @($policy.residencyTelemetryChecks.requiredCrossfadePreparedStateFieldApplications)) {
    $fieldPathRegex = "$($requiredField.pathRegex)"
    if ([string]::IsNullOrWhiteSpace($fieldPathRegex)) {
        continue
    }

    $covered = $false
    foreach ($requiredConsumer in @($policy.residencyTelemetryChecks.requiredCrossfadePreparedSnapshotConsumers)) {
        $consumerPathRegex = "$($requiredConsumer.pathRegex)"
        if ([string]::IsNullOrWhiteSpace($consumerPathRegex)) {
            continue
        }
        if ($consumerPathRegex -eq $fieldPathRegex) {
            $covered = $true
            break
        }
    }

    if (-not $covered) {
        $crossfadePreparedFieldToConsumerCoverageViolations += [ordered]@{
            file = '<policy>'
            line = 0
            snippet = $fieldPathRegex
            checkId = 'CI-CROSSFADE-013'
            message = 'required prepared-field application pathRegex is not covered by required prepared-snapshot consumers'
        }
    }
}

$crossfadePreparedPolicyDuplicatePathViolations = @()

$requiredConsumerPathCounts = @{}
foreach ($requiredConsumer in @($policy.residencyTelemetryChecks.requiredCrossfadePreparedSnapshotConsumers)) {
    $pathRegex = "$($requiredConsumer.pathRegex)"
    if ([string]::IsNullOrWhiteSpace($pathRegex)) {
        continue
    }
    if (-not $requiredConsumerPathCounts.ContainsKey($pathRegex)) {
        $requiredConsumerPathCounts[$pathRegex] = 0
    }
    $requiredConsumerPathCounts[$pathRegex]++
}
foreach ($key in $requiredConsumerPathCounts.Keys) {
    if ($requiredConsumerPathCounts[$key] -gt 1) {
        $crossfadePreparedPolicyDuplicatePathViolations += [ordered]@{
            file = '<policy>'
            line = 0
            snippet = $key
            checkId = 'CI-CROSSFADE-014'
            message = 'duplicate pathRegex found in requiredCrossfadePreparedSnapshotConsumers'
        }
    }
}

$requiredFieldPathCounts = @{}
foreach ($requiredField in @($policy.residencyTelemetryChecks.requiredCrossfadePreparedStateFieldApplications)) {
    $pathRegex = "$($requiredField.pathRegex)"
    if ([string]::IsNullOrWhiteSpace($pathRegex)) {
        continue
    }
    if (-not $requiredFieldPathCounts.ContainsKey($pathRegex)) {
        $requiredFieldPathCounts[$pathRegex] = 0
    }
    $requiredFieldPathCounts[$pathRegex]++
}
foreach ($key in $requiredFieldPathCounts.Keys) {
    if ($requiredFieldPathCounts[$key] -gt 1) {
        $crossfadePreparedPolicyDuplicatePathViolations += [ordered]@{
            file = '<policy>'
            line = 0
            snippet = $key
            checkId = 'CI-CROSSFADE-014'
            message = 'duplicate pathRegex found in requiredCrossfadePreparedStateFieldApplications'
        }
    }
}

$crossfadePreparedAllowlistDuplicatePathViolations = @()

$consumerAllowlistPathCounts = @{}
foreach ($allowedScope in @($policy.residencyTelemetryChecks.crossfadePreparedSnapshotConsumerAllowedScopes)) {
    $pathRegex = "$($allowedScope.pathRegex)"
    if ([string]::IsNullOrWhiteSpace($pathRegex)) {
        continue
    }
    if (-not $consumerAllowlistPathCounts.ContainsKey($pathRegex)) {
        $consumerAllowlistPathCounts[$pathRegex] = 0
    }
    $consumerAllowlistPathCounts[$pathRegex]++
}
foreach ($key in $consumerAllowlistPathCounts.Keys) {
    if ($consumerAllowlistPathCounts[$key] -gt 1) {
        $crossfadePreparedAllowlistDuplicatePathViolations += [ordered]@{
            file = '<policy>'
            line = 0
            snippet = $key
            checkId = 'CI-CROSSFADE-015'
            message = 'duplicate pathRegex found in crossfadePreparedSnapshotConsumerAllowedScopes'
        }
    }
}

$fieldAllowlistPathCounts = @{}
foreach ($allowedScope in @($policy.residencyTelemetryChecks.crossfadePreparedStateFieldUsageAllowedScopes)) {
    $pathRegex = "$($allowedScope.pathRegex)"
    if ([string]::IsNullOrWhiteSpace($pathRegex)) {
        continue
    }
    if (-not $fieldAllowlistPathCounts.ContainsKey($pathRegex)) {
        $fieldAllowlistPathCounts[$pathRegex] = 0
    }
    $fieldAllowlistPathCounts[$pathRegex]++
}
foreach ($key in $fieldAllowlistPathCounts.Keys) {
    if ($fieldAllowlistPathCounts[$key] -gt 1) {
        $crossfadePreparedAllowlistDuplicatePathViolations += [ordered]@{
            file = '<policy>'
            line = 0
            snippet = $key
            checkId = 'CI-CROSSFADE-015'
            message = 'duplicate pathRegex found in crossfadePreparedStateFieldUsageAllowedScopes'
        }
    }
}

$crossfadePreparedAllowlistCoverageViolations = @()

$consumerAllowlistPaths = @{}
foreach ($allowedScope in @($policy.residencyTelemetryChecks.crossfadePreparedSnapshotConsumerAllowedScopes)) {
    $pathRegex = "$($allowedScope.pathRegex)"
    if ([string]::IsNullOrWhiteSpace($pathRegex)) {
        continue
    }
    $consumerAllowlistPaths[$pathRegex] = $true
}

$fieldAllowlistPaths = @{}
foreach ($allowedScope in @($policy.residencyTelemetryChecks.crossfadePreparedStateFieldUsageAllowedScopes)) {
    $pathRegex = "$($allowedScope.pathRegex)"
    if ([string]::IsNullOrWhiteSpace($pathRegex)) {
        continue
    }
    $fieldAllowlistPaths[$pathRegex] = $true
}

foreach ($consumerPath in $consumerAllowlistPaths.Keys) {
    if (-not $fieldAllowlistPaths.ContainsKey($consumerPath)) {
        $crossfadePreparedAllowlistCoverageViolations += [ordered]@{
            file = '<policy>'
            line = 0
            snippet = $consumerPath
            checkId = 'CI-CROSSFADE-016'
            message = 'consumer allowlist pathRegex is not covered by state-field allowlist'
        }
    }
}

foreach ($fieldPath in $fieldAllowlistPaths.Keys) {
    if (-not $consumerAllowlistPaths.ContainsKey($fieldPath)) {
        $crossfadePreparedAllowlistCoverageViolations += [ordered]@{
            file = '<policy>'
            line = 0
            snippet = $fieldPath
            checkId = 'CI-CROSSFADE-016'
            message = 'state-field allowlist pathRegex is not covered by consumer allowlist'
        }
    }
}

$crossfadePreparedAllowlistOverreachViolations = @()
$requiredCrossfadePathUniverse = @{}

foreach ($requiredConsumer in @($policy.residencyTelemetryChecks.requiredCrossfadePreparedSnapshotConsumers)) {
    $pathRegex = "$($requiredConsumer.pathRegex)"
    if (-not [string]::IsNullOrWhiteSpace($pathRegex)) {
        $requiredCrossfadePathUniverse[$pathRegex] = $true
    }
}
foreach ($requiredSnapshot in @($policy.residencyTelemetryChecks.requiredCrossfadePreparedSnapshotApplications)) {
    $pathRegex = "$($requiredSnapshot.pathRegex)"
    if (-not [string]::IsNullOrWhiteSpace($pathRegex)) {
        $requiredCrossfadePathUniverse[$pathRegex] = $true
    }
}
foreach ($requiredField in @($policy.residencyTelemetryChecks.requiredCrossfadePreparedStateFieldApplications)) {
    $pathRegex = "$($requiredField.pathRegex)"
    if (-not [string]::IsNullOrWhiteSpace($pathRegex)) {
        $requiredCrossfadePathUniverse[$pathRegex] = $true
    }
}

foreach ($consumerPath in $consumerAllowlistPaths.Keys) {
    if (-not $requiredCrossfadePathUniverse.ContainsKey($consumerPath)) {
        $crossfadePreparedAllowlistOverreachViolations += [ordered]@{
            file = '<policy>'
            line = 0
            snippet = $consumerPath
            checkId = 'CI-CROSSFADE-017'
            message = 'consumer allowlist pathRegex is not declared by required crossfade prepared-* applications'
        }
    }
}

foreach ($fieldPath in $fieldAllowlistPaths.Keys) {
    if (-not $requiredCrossfadePathUniverse.ContainsKey($fieldPath)) {
        $crossfadePreparedAllowlistOverreachViolations += [ordered]@{
            file = '<policy>'
            line = 0
            snippet = $fieldPath
            checkId = 'CI-CROSSFADE-017'
            message = 'state-field allowlist pathRegex is not declared by required crossfade prepared-* applications'
        }
    }
}

$residencyPolicyOrphanPathViolations = @()

$residencyPolicyPathEntries = @()
$residencyPolicyPathEntries += @($policy.residencyContainerAllowlist | ForEach-Object { [ordered]@{ pathRegex = $_.pathRegex; source = 'residencyContainerAllowlist.pathRegex' } })
$residencyPolicyPathEntries += @($policy.telemetryCounterOwners | ForEach-Object { [ordered]@{ pathRegex = $_.allowedPathRegex; source = 'telemetryCounterOwners.allowedPathRegex' } })
$residencyPolicyPathEntries += @($policy.residencyTelemetryChecks.crossfadeRtForbiddenWriteScopes | ForEach-Object { [ordered]@{ pathRegex = $_.pathRegex; source = 'crossfadeRtForbiddenWriteScopes.pathRegex' } })
$residencyPolicyPathEntries += @($policy.residencyTelemetryChecks.requiredCrossfadeAuthorityApplications | ForEach-Object { [ordered]@{ pathRegex = $_.pathRegex; source = 'requiredCrossfadeAuthorityApplications.pathRegex' } })
$residencyPolicyPathEntries += @($policy.residencyTelemetryChecks.crossfadeAuthorityAllowedWriteScopes | ForEach-Object { [ordered]@{ pathRegex = $_.pathRegex; source = 'crossfadeAuthorityAllowedWriteScopes.pathRegex' } })
$residencyPolicyPathEntries += @($policy.residencyTelemetryChecks.crossfadeAtomicReadAllowlist | ForEach-Object { [ordered]@{ pathRegex = $_.pathRegex; source = 'crossfadeAtomicReadAllowlist.pathRegex' } })
$residencyPolicyPathEntries += @($policy.residencyTelemetryChecks.requiredCrossfadePreparedSnapshotApplications | ForEach-Object { [ordered]@{ pathRegex = $_.pathRegex; source = 'requiredCrossfadePreparedSnapshotApplications.pathRegex' } })
$residencyPolicyPathEntries += @($policy.residencyTelemetryChecks.requiredCrossfadePreparedSnapshotConsumers | ForEach-Object { [ordered]@{ pathRegex = $_.pathRegex; source = 'requiredCrossfadePreparedSnapshotConsumers.pathRegex' } })
$residencyPolicyPathEntries += @($policy.residencyTelemetryChecks.crossfadePreparedSnapshotConsumerAllowedScopes | ForEach-Object { [ordered]@{ pathRegex = $_.pathRegex; source = 'crossfadePreparedSnapshotConsumerAllowedScopes.pathRegex' } })
$residencyPolicyPathEntries += @($policy.residencyTelemetryChecks.crossfadePreparedStateFieldUsageAllowedScopes | ForEach-Object { [ordered]@{ pathRegex = $_.pathRegex; source = 'crossfadePreparedStateFieldUsageAllowedScopes.pathRegex' } })
$residencyPolicyPathEntries += @($policy.residencyTelemetryChecks.requiredCrossfadePreparedStateFieldApplications | ForEach-Object { [ordered]@{ pathRegex = $_.pathRegex; source = 'requiredCrossfadePreparedStateFieldApplications.pathRegex' } })
$residencyPolicyPathEntries += @($policy.residencyTelemetryChecks.forbiddenTelemetryAuthorityScopes | ForEach-Object { [ordered]@{ pathRegex = $_.pathRegex; source = 'forbiddenTelemetryAuthorityScopes.pathRegex' } })
$residencyPolicyPathEntries += @($policy.residencyTelemetryChecks.requiredTelemetryApplications | ForEach-Object { [ordered]@{ pathRegex = $_.pathRegex; source = 'requiredTelemetryApplications.pathRegex' } })

foreach ($entry in @($residencyPolicyPathEntries)) {
    $pathRegex = "$($entry.pathRegex)"
    if ([string]::IsNullOrWhiteSpace($pathRegex)) {
        continue
    }

    if (-not (Test-RegexMatchesAnyPath -PathRegex $pathRegex -RelativePaths $sourceRelativePaths)) {
        $residencyPolicyOrphanPathViolations += [ordered]@{
            file    = 'policy'
            line    = 0
            snippet = $pathRegex
            checkId = 'CI-RESIDENCY-005'
            message = "$($entry.source) matches no source files"
        }
    }
}

$violations = @($residencyViolations + $residencyOwnershipViolations + $residencyAuthorityViolations + $residencyContractDuplicateViolations + $residencyPolicyOrphanPathViolations + $telemetryViolations + $telemetryOwnerCoverageViolations + $requiredTelemetryPatternViolations + $requiredTelemetryApplicationViolations + $telemetryImmediateSyncViolations + $telemetryImmediateSyncPolicyCoverageViolations + $requiredTelemetryOwnerScopeCoverageViolations + $telemetryOwnerRequiredApplicationCoverageViolations + $telemetryPolicyDuplicateViolations + $forbiddenTelemetryAuthorityViolations + $crossfadeRtWriteViolations + $requiredCrossfadeAuthorityApplicationViolations + $crossfadeAuthorityScopeViolations + $crossfadeAtomicReadViolations + $requiredCrossfadePreparedSnapshotViolations + $requiredCrossfadePreparedSnapshotConsumerViolations + $crossfadePreparedConsumerPolicyCoverageViolations + $crossfadePreparedSnapshotScopeViolations + $crossfadePreparedStateFieldScopeViolations + $requiredCrossfadePreparedStateFieldApplicationViolations + $crossfadePreparedFieldPolicyCoverageViolations + $crossfadePreparedConsumerToFieldCoverageViolations + $crossfadePreparedFieldToConsumerCoverageViolations + $crossfadePreparedPolicyDuplicatePathViolations + $crossfadePreparedAllowlistDuplicatePathViolations + $crossfadePreparedAllowlistCoverageViolations + $crossfadePreparedAllowlistOverreachViolations)
$report = [ordered]@{
    schema = 'isr_v73_residency_telemetry_report_v1'
    generatedAt = (Get-Date -Format 'o')
    mode = $effectiveMode
    checks = @('CI-RESIDENCY-001', 'CI-RESIDENCY-002', 'CI-RESIDENCY-003', 'CI-RESIDENCY-004', 'CI-RESIDENCY-005', 'CI-TELEMETRY-001', 'CI-TELEMETRY-002', 'CI-TELEMETRY-003', 'CI-TELEMETRY-004', 'CI-TELEMETRY-005', 'CI-TELEMETRY-006', 'CI-TELEMETRY-007', 'CI-TELEMETRY-008', 'CI-TELEMETRY-009', 'CI-TELEMETRY-010', 'CI-CROSSFADE-001', 'CI-CROSSFADE-002', 'CI-CROSSFADE-003', 'CI-CROSSFADE-004', 'CI-CROSSFADE-005', 'CI-CROSSFADE-006', 'CI-CROSSFADE-007', 'CI-CROSSFADE-008', 'CI-CROSSFADE-009', 'CI-CROSSFADE-010', 'CI-CROSSFADE-011', 'CI-CROSSFADE-012', 'CI-CROSSFADE-013', 'CI-CROSSFADE-014', 'CI-CROSSFADE-015', 'CI-CROSSFADE-016', 'CI-CROSSFADE-017')
    violationCount = $violations.Count
    violations = $violations
}

$report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $reportPath -Encoding UTF8
Write-Host "[INFO] wrote $reportPath"

if ($violations.Count -gt 0) {
    foreach ($v in $violations) {
        Write-Host "[WARN] [$($v.checkId)] $($v.file):$($v.line) $($v.message)"
    }

    if ($effectiveMode -eq 'fail') {
        throw "ISR v7.3 residency/telemetry checks failed. violations=$($violations.Count)"
    }

    Write-Host "[WARN] ISR v7.3 residency/telemetry checks completed in warn mode. violations=$($violations.Count)"
    exit 0
}

Write-Host '[PASS] ISR v7.3 residency/telemetry checks passed'
exit 0
