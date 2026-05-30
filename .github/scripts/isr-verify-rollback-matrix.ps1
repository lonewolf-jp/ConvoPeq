param(
    [string]$MatrixPath = '.github/isr-rollback-compatibility-matrix.json'
)

$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$evidenceDir = Join-Path $repoRoot 'evidence'
$reportPath = Join-Path $evidenceDir 'rollback_compatibility_report.json'
$retireTimelinePath = Join-Path $evidenceDir 'retire_timeline.json'
$resolvedMatrixPath = if ([System.IO.Path]::IsPathRooted($MatrixPath)) { $MatrixPath } else { Join-Path $repoRoot $MatrixPath }
$metricGovernancePath = Join-Path $repoRoot '.github/isr-metric-governance.json'
$retireRuntimeHeaderPath = Join-Path $repoRoot 'src/audioengine/ISRRetireRuntimeEx.h'
$retireRuntimeCppPath = Join-Path $repoRoot 'src/audioengine/ISRRetireRuntimeEx.cpp'

if (-not (Test-Path $resolvedMatrixPath)) {
    throw "Missing rollback compatibility matrix: $resolvedMatrixPath"
}
if (-not (Test-Path $metricGovernancePath)) {
    throw "Missing metric governance policy: $metricGovernancePath"
}
if (-not (Test-Path $retireRuntimeHeaderPath)) {
    throw "Missing retire runtime header: $retireRuntimeHeaderPath"
}
if (-not (Test-Path $retireRuntimeCppPath)) {
    throw "Missing retire runtime source: $retireRuntimeCppPath"
}
if (-not (Test-Path $evidenceDir)) {
    New-Item -Path $evidenceDir -ItemType Directory | Out-Null
}
if (-not (Test-Path $retireTimelinePath)) {
    throw "Missing retire timeline evidence: $retireTimelinePath"
}

$matrix = Get-Content -LiteralPath $resolvedMatrixPath -Raw -Encoding UTF8 | ConvertFrom-Json
$metricGovernance = Get-Content -LiteralPath $metricGovernancePath -Raw -Encoding UTF8 | ConvertFrom-Json
$retireRuntimeHeaderText = Get-Content -LiteralPath $retireRuntimeHeaderPath -Raw -Encoding UTF8
$retireRuntimeCppText = Get-Content -LiteralPath $retireRuntimeCppPath -Raw -Encoding UTF8
$retireTimeline = Get-Content -LiteralPath $retireTimelinePath -Raw -Encoding UTF8 | ConvertFrom-Json
if ($matrix.schema -ne 'rollback_compatibility_matrix_v1') {
    throw "Unexpected rollback matrix schema: $($matrix.schema)"
}

foreach ($field in @('owner', 'issue', 'rationale', 'expiry')) {
    if ($null -eq $matrix.$field -or [string]::IsNullOrWhiteSpace("$($matrix.$field)")) {
        throw "rollback matrix missing required field: $field"
    }
}

$matrixExpiry = [datetime]::ParseExact("$($matrix.expiry)", 'yyyy-MM-dd', [System.Globalization.CultureInfo]::InvariantCulture)
if ((Get-Date).Date -gt $matrixExpiry.Date) {
    throw "rollback matrix expired: expiry=$($matrix.expiry) owner=$($matrix.owner) issue=$($matrix.issue)"
}

if ($metricGovernance.schema -ne 'metric_governance_v1') {
    throw "Unexpected metric governance schema: $($metricGovernance.schema)"
}
if ($retireTimeline.schema -ne 'retire_timeline_v1' -and $retireTimeline.schema -ne 'retire_timeline_v2') {
    throw "Unexpected retire timeline schema: $($retireTimeline.schema)"
}

if ([string]::IsNullOrWhiteSpace("$($matrix.globalFlag)")) {
    throw 'rollback matrix missing globalFlag'
}
if (-not $matrix.subsystemFlags -or $matrix.subsystemFlags.Count -lt 3) {
    throw 'rollback matrix requires at least 3 subsystem flags'
}
if (-not $matrix.compatibility -or $matrix.compatibility.Count -eq 0) {
    throw 'rollback matrix compatibility scenarios must be non-empty'
}

$today = (Get-Date).Date
$violations = New-Object System.Collections.Generic.List[string]
$knownFlags = New-Object System.Collections.Generic.HashSet[string]
$actionCoverage = New-Object 'System.Collections.Generic.Dictionary[string,int]'
$knownFlags.Add("$($matrix.globalFlag)") | Out-Null

foreach ($entry in $matrix.subsystemFlags) {
    foreach ($field in @('id', 'flag', 'owner', 'issue', 'rationale', 'expiry')) {
        if ($null -eq $entry.$field -or [string]::IsNullOrWhiteSpace("$($entry.$field)")) {
            throw "subsystem flag entry missing required field: $field"
        }
    }

    $knownFlags.Add("$($entry.flag)") | Out-Null
    $actionCoverage["$($entry.flag)"] = 0
    $expiry = [datetime]::ParseExact("$($entry.expiry)", 'yyyy-MM-dd', [System.Globalization.CultureInfo]::InvariantCulture)
    if ($today -gt $expiry.Date) {
        $violations.Add("Expired subsystem rollback flag entry: id=$($entry.id) flag=$($entry.flag) expiry=$($entry.expiry) owner=$($entry.owner) issue=$($entry.issue)")
    }
}

if (-not ($retireTimeline.PSObject.Properties.Name -contains 'rollbackFlags')) {
    $violations.Add('retire_timeline missing rollbackFlags object')
}
else {
    foreach ($flagName in @('global', 'publicationOnly', 'crossfadeOnly', 'retirePathOnly')) {
        if (-not ($retireTimeline.rollbackFlags.PSObject.Properties.Name -contains $flagName)) {
            $violations.Add("retire_timeline.rollbackFlags missing field: $flagName")
            continue
        }

        $flagValue = $retireTimeline.rollbackFlags.$flagName
        if ($flagValue -isnot [bool]) {
            $violations.Add("retire_timeline.rollbackFlags.$flagName must be boolean")
        }
    }

    if (-not ($retireTimeline.PSObject.Properties.Name -contains 'rollbackReady')) {
        $violations.Add('retire_timeline missing rollbackReady field')
    }
    elseif ($retireTimeline.rollbackReady -isnot [bool]) {
        $violations.Add('retire_timeline.rollbackReady must be boolean')
    }
    elseif (($retireTimeline.rollbackFlags.global -is [bool]) -and ($retireTimeline.rollbackFlags.retirePathOnly -is [bool])) {
        $expectedRollbackReady = ($retireTimeline.rollbackFlags.global -and $retireTimeline.rollbackFlags.retirePathOnly)
        if ([bool]$retireTimeline.rollbackReady -ne $expectedRollbackReady) {
            $violations.Add('retire_timeline rollbackReady invariant mismatch: expected rollbackFlags.global && rollbackFlags.retirePathOnly')
        }
    }
}

$requiredHeaderTokens = @(
    'struct RollbackFlagDescriptor',
    'setRollbackFlags\(',
    'describeRollbackFlags\(',
    'rollbackGlobalEnabled_',
    'rollbackPublicationOnlyEnabled_',
    'rollbackCrossfadeOnlyEnabled_',
    'rollbackRetirePathOnlyEnabled_'
)

foreach ($token in $requiredHeaderTokens) {
    if ($retireRuntimeHeaderText -notmatch $token) {
        $violations.Add("RetireRuntimeEx header missing rollback hierarchy token: token=$token")
    }
}

$requiredCppTokens = @(
    'readEnvFlag\("ISR_ROLLBACK_GLOBAL"',
    'readEnvFlag\("ISR_ROLLBACK_PUBLICATION_ONLY"',
    'readEnvFlag\("ISR_ROLLBACK_CROSSFADE_ONLY"',
    'readEnvFlag\("ISR_ROLLBACK_RETIRE_PATH_ONLY"',
    'describeRollbackFlags\(',
    'rollbackFlags'
)

foreach ($token in $requiredCppTokens) {
    if ($retireRuntimeCppText -notmatch $token) {
        $violations.Add("RetireRuntimeEx source missing rollback hierarchy token: token=$token")
    }
}

$scenarioReports = New-Object System.Collections.Generic.List[object]
foreach ($scenario in $matrix.compatibility) {
    foreach ($field in @('scenario', 'requiredFlags', 'owner', 'issue', 'rationale', 'expiry')) {
        if ($null -eq $scenario.$field -or [string]::IsNullOrWhiteSpace("$($scenario.$field)")) {
            throw "compatibility scenario missing required field: $field"
        }
    }

    $expiry = [datetime]::ParseExact("$($scenario.expiry)", 'yyyy-MM-dd', [System.Globalization.CultureInfo]::InvariantCulture)
    $expired = $today -gt $expiry.Date
    if ($expired) {
        $violations.Add("Expired rollback compatibility scenario: scenario=$($scenario.scenario) expiry=$($scenario.expiry) owner=$($scenario.owner) issue=$($scenario.issue)")
    }

    $missingFlags = @()
    foreach ($flag in $scenario.requiredFlags) {
        if (-not $knownFlags.Contains("$flag")) {
            $missingFlags += "$flag"
        }
    }

    if ($missingFlags.Count -gt 0) {
        $violations.Add("Rollback compatibility scenario references unknown flags: scenario=$($scenario.scenario) missing=$($missingFlags -join ',')")
    }

    $scenarioReports.Add([ordered]@{
            scenario      = "$($scenario.scenario)"
            requiredFlags = $scenario.requiredFlags
            expired       = $expired
            missingFlags  = $missingFlags
        }) | Out-Null
}

if (-not $metricGovernance.metrics -or $metricGovernance.metrics.Count -eq 0) {
    throw 'metric governance requires at least one metric entry'
}

$rollbackTokenRegex = [regex]'ISR_ROLLBACK_[A-Z0-9_]+'
foreach ($metric in $metricGovernance.metrics) {
    $actionText = "$($metric.action)"
    if ([string]::IsNullOrWhiteSpace($actionText)) {
        continue
    }

    $tokenMatches = $rollbackTokenRegex.Matches($actionText)
    foreach ($match in $tokenMatches) {
        $flagToken = "$($match.Value)"
        if (-not $knownFlags.Contains($flagToken)) {
            $violations.Add("Metric action references unknown rollback flag: metric=$($metric.id) flag=$flagToken")
            continue
        }

        if ($actionCoverage.ContainsKey($flagToken)) {
            $actionCoverage[$flagToken] = $actionCoverage[$flagToken] + 1
        }
    }
}

foreach ($entry in $matrix.subsystemFlags) {
    $flag = "$($entry.flag)"
    if (-not $actionCoverage.ContainsKey($flag) -or $actionCoverage[$flag] -lt 1) {
        $violations.Add("Subsystem rollback flag lacks metric action coverage: flag=$flag issue=$($entry.issue)")
    }
}

$report = [ordered]@{
    schema               = 'rollback_compatibility_report_v1'
    generatedAt          = (Get-Date -Format 'o')
    matrixPath           = $resolvedMatrixPath
    matrixOwner          = "$($matrix.owner)"
    matrixIssue          = "$($matrix.issue)"
    matrixExpiry         = "$($matrix.expiry)"
    globalFlag           = "$($matrix.globalFlag)"
    subsystemFlagCount   = $matrix.subsystemFlags.Count
    scenarioCount        = $matrix.compatibility.Count
    metricActionCoverage = $actionCoverage
    scenarios            = $scenarioReports
    violations           = $violations
}

$reportJson = $report | ConvertTo-Json -Depth 8
Set-Content -LiteralPath $reportPath -Value $reportJson -Encoding UTF8
Write-Host "[INFO] rollback compatibility report written: $reportPath"

if ($violations.Count -gt 0) {
    foreach ($violation in $violations) {
        Write-Host "[ERROR] $violation"
    }
    throw "Rollback compatibility matrix violations detected. count=$($violations.Count)"
}

Write-Host '[PASS] rollback compatibility matrix gate verified'
