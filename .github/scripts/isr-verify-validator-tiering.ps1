$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$tierRunnerPath = Join-Path $repoRoot '.github\scripts\isr-run-tiered-verification.ps1'
$workflowPath = Join-Path $repoRoot '.github\workflows\isr-verification.yml'
$planPath = Join-Path $repoRoot 'doc\work\bridge_runtime_migration_plan.md'
$policyPath = Join-Path $repoRoot '.github\isr-validator-tiering-policy.json'
$sourceRoot = Join-Path $repoRoot 'src'
$evidenceDir = Join-Path $repoRoot 'evidence'
$reportPath = Join-Path $evidenceDir 'validator_tiering_report.json'

foreach ($path in @($tierRunnerPath, $workflowPath, $planPath, $policyPath, $sourceRoot)) {
    if (-not (Test-Path $path)) {
        throw "Missing required file: $path"
    }
}

if (-not (Test-Path $evidenceDir)) {
    New-Item -Path $evidenceDir -ItemType Directory | Out-Null
}

$tierRunnerText = Get-Content -LiteralPath $tierRunnerPath -Raw -Encoding UTF8
$workflowText = Get-Content -LiteralPath $workflowPath -Raw -Encoding UTF8
$planText = Get-Content -LiteralPath $planPath -Raw -Encoding UTF8
$policy = Get-Content -LiteralPath $policyPath -Raw -Encoding UTF8 | ConvertFrom-Json
$sourceFiles = Get-ChildItem -LiteralPath $sourceRoot -Recurse -File -Include *.h, *.hpp, *.cpp, *.cc, *.cxx -ErrorAction SilentlyContinue

$violations = New-Object System.Collections.Generic.List[string]

if ($tierRunnerText -notmatch "ValidateSet\('smoke', 'standard', 'exhaustive'\)") {
    $violations.Add('Tier runner must define smoke/standard/exhaustive ValidateSet')
}

foreach ($field in @('owner', 'issue', 'rationale', 'expiry')) {
    if ($null -eq $policy.$field -or [string]::IsNullOrWhiteSpace("$($policy.$field)")) {
        $violations.Add("Validator tiering policy missing required field: $field")
    }
}

if ("$($policy.schema)" -ne 'isr_validator_tiering_policy_v1') {
    $violations.Add("Validator tiering policy schema mismatch: expected=isr_validator_tiering_policy_v1 actual=$($policy.schema)")
}

try {
    $expiryUtc = ([datetimeoffset]::Parse($policy.expiry)).UtcDateTime
    if ($expiryUtc -le (Get-Date).ToUniversalTime()) {
        $violations.Add("Validator tiering policy expired: $($policy.expiry)")
    }
}
catch {
    $violations.Add("Validator tiering policy expiry is not parseable ISO-8601: $($policy.expiry)")
}

if ($null -eq $policy.tiers) {
    $violations.Add('Validator tiering policy missing tiers object')
}
else {
    if ([string]$policy.tiers.smoke -ne 'pr') {
        $violations.Add("Validator tiering policy tiers.smoke must be 'pr' but was '$($policy.tiers.smoke)'")
    }
    if ([string]$policy.tiers.standard -ne 'nightly') {
        $violations.Add("Validator tiering policy tiers.standard must be 'nightly' but was '$($policy.tiers.standard)'")
    }
    if ([string]$policy.tiers.exhaustive -ne 'weekly') {
        $violations.Add("Validator tiering policy tiers.exhaustive must be 'weekly' but was '$($policy.tiers.exhaustive)'")
    }
}

if ($null -eq $policy.workflowSchedule) {
    $violations.Add('Validator tiering policy missing workflowSchedule object')
}
else {
    foreach ($field in @('nightlyCron', 'weeklyCron', 'nightlyTier', 'weeklyTier')) {
        if ($null -eq $policy.workflowSchedule.$field -or [string]::IsNullOrWhiteSpace("$($policy.workflowSchedule.$field)")) {
            $violations.Add("Validator tiering policy workflowSchedule missing required field: $field")
        }
    }

    $nightlyCron = "$($policy.workflowSchedule.nightlyCron)"
    $weeklyCron = "$($policy.workflowSchedule.weeklyCron)"
    $nightlyTier = "$($policy.workflowSchedule.nightlyTier)"
    $weeklyTier = "$($policy.workflowSchedule.weeklyTier)"

    if (($nightlyTier -ne 'smoke' -and $nightlyTier -ne 'standard' -and $nightlyTier -ne 'exhaustive') -or
        ($weeklyTier -ne 'smoke' -and $weeklyTier -ne 'standard' -and $weeklyTier -ne 'exhaustive')) {
        $violations.Add("Validator tiering policy workflowSchedule tier values must be one of smoke/standard/exhaustive: nightlyTier=$nightlyTier weeklyTier=$weeklyTier")
    }

    if ($nightlyCron -eq $weeklyCron) {
        $violations.Add('Validator tiering policy workflowSchedule requires distinct nightlyCron and weeklyCron')
    }

    $workflowCrons = @([regex]::Matches($workflowText, "(?m)^\s*-\s*cron:\s*'(?<cron>[^']+)'\s*$") | ForEach-Object { $_.Groups['cron'].Value })
    if ($workflowCrons.Count -eq 0) {
        $violations.Add('Workflow schedule contract mismatch: no cron entries found in workflow')
    }
    else {
        $workflowCronSet = New-Object 'System.Collections.Generic.HashSet[string]'
        foreach ($cron in $workflowCrons) {
            [void]$workflowCronSet.Add($cron)
        }

        if (-not $workflowCronSet.Contains($nightlyCron)) {
            $violations.Add("Workflow schedule contract mismatch: nightlyCron missing in workflow schedule: $nightlyCron")
        }
        if (-not $workflowCronSet.Contains($weeklyCron)) {
            $violations.Add("Workflow schedule contract mismatch: weeklyCron missing in workflow schedule: $weeklyCron")
        }

        if ($workflowCronSet.Count -ne 2 -or $workflowCrons.Count -ne 2) {
            $violations.Add("Workflow schedule contract mismatch: expected exactly two cron entries (nightly/weekly) but found count=$($workflowCrons.Count)")
        }
    }

    $weeklyScheduleCondition = '"${{ github.event.schedule }}" -eq ''' + $weeklyCron + ''''
    $weeklyScheduleLiteralBranch = $workflowText.Contains($weeklyScheduleCondition)
    $weeklySchedulePolicyBranch = $workflowText.Contains('"${{ github.event.schedule }}" -eq $validatorTieringScheduleContract.weeklyCron')
    if (-not ($weeklyScheduleLiteralBranch -or $weeklySchedulePolicyBranch)) {
        $violations.Add("Workflow schedule contract mismatch: weekly schedule branch missing for cron=$weeklyCron")
    }

    $weeklyTierLiteralAssignment = $workflowText.Contains("`$tier = '$weeklyTier'")
    $weeklyTierPolicyAssignment = $workflowText.Contains('$tier = $validatorTieringScheduleContract.weeklyTier')
    if (-not ($weeklyTierLiteralAssignment -or $weeklyTierPolicyAssignment)) {
        $violations.Add("Workflow schedule contract mismatch: weekly tier assignment missing for tier=$weeklyTier")
    }

    $nightlyTierLiteralAssignment = $workflowText.Contains("`$tier = '$nightlyTier'")
    $nightlyTierPolicyAssignment = $workflowText.Contains('$tier = $validatorTieringScheduleContract.nightlyTier')
    if (-not ($nightlyTierLiteralAssignment -or $nightlyTierPolicyAssignment)) {
        $violations.Add("Workflow schedule contract mismatch: nightly tier assignment missing for tier=$nightlyTier")
    }

    if (-not $workflowText.Contains('Unknown workflow schedule cron:')) {
        $violations.Add('Workflow schedule contract mismatch: unknown schedule cron must fail-closed')
    }
}

if ($null -eq $policy.slaHours) {
    $violations.Add('Validator tiering policy missing slaHours object')
}
else {
    if ([int]$policy.slaHours.hbViolation -ne 24) {
        $violations.Add("Validator tiering policy slaHours.hbViolation must be 24 but was '$($policy.slaHours.hbViolation)'")
    }
    if ([int]$policy.slaHours.payloadMismatch -ne 72) {
        $violations.Add("Validator tiering policy slaHours.payloadMismatch must be 72 but was '$($policy.slaHours.payloadMismatch)'")
    }
}

function Resolve-ArtifactTimestampUtc {
    param(
        [Parameter(Mandatory = $true)][object]$Artifact,
        [Parameter(Mandatory = $true)][string]$ArtifactPath
    )

    if ($null -ne $Artifact.generatedAtNs -and [long]$Artifact.generatedAtNs -gt 0) {
        $generatedAtMs = [long][math]::Floor(([double]$Artifact.generatedAtNs) / 1000000.0)
        return ([DateTimeOffset]::FromUnixTimeMilliseconds($generatedAtMs)).UtcDateTime
    }

    if (-not [string]::IsNullOrWhiteSpace("$($Artifact.generatedAt)")) {
        return ([datetimeoffset]::Parse("$($Artifact.generatedAt)")).UtcDateTime
    }

    throw "Artifact timestamp missing (generatedAtNs/generatedAt): $ArtifactPath"
}

$slaFreshness = New-Object System.Collections.Generic.List[object]
$nowUtc = (Get-Date).ToUniversalTime()

$slaChecks = @(
    [ordered]@{
        key            = 'hbViolation'
        artifactPath   = (Join-Path $repoRoot 'evidence\hb_violation_report.json')
        expectedSchema = 'hb_violation_report_v1'
        maxAgeHours    = [int]$policy.slaHours.hbViolation
    },
    [ordered]@{
        key            = 'payloadMismatch'
        artifactPath   = (Join-Path $repoRoot 'evidence\payload_tier_report.json')
        expectedSchema = 'payload_tier_report_v1'
        maxAgeHours    = [int]$policy.slaHours.payloadMismatch
    }
)

foreach ($slaCheck in $slaChecks) {
    if (-not (Test-Path -LiteralPath $slaCheck.artifactPath)) {
        $violations.Add("SLA artifact missing: key=$($slaCheck.key) path=$($slaCheck.artifactPath)")
        $slaFreshness.Add([ordered]@{
                key            = $slaCheck.key
                artifactPath   = $slaCheck.artifactPath
                expectedSchema = $slaCheck.expectedSchema
                maxAgeHours    = $slaCheck.maxAgeHours
                present        = $false
                withinSla      = $false
            }) | Out-Null
        continue
    }

    $artifact = Get-Content -LiteralPath $slaCheck.artifactPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $schema = "$($artifact.schema)"
    if ($schema -ne $slaCheck.expectedSchema) {
        $violations.Add("SLA artifact schema mismatch: key=$($slaCheck.key) expected=$($slaCheck.expectedSchema) actual=$schema")
    }

    try {
        $generatedAtUtc = Resolve-ArtifactTimestampUtc -Artifact $artifact -ArtifactPath $slaCheck.artifactPath
        $ageHours = [Math]::Round(($nowUtc - $generatedAtUtc).TotalHours, 2)
        $withinSla = ($ageHours -le [double]$slaCheck.maxAgeHours)

        if (-not $withinSla) {
            $violations.Add("SLA breach: key=$($slaCheck.key) ageHours=$ageHours maxAgeHours=$($slaCheck.maxAgeHours)")
        }

        $slaFreshness.Add([ordered]@{
                key            = $slaCheck.key
                artifactPath   = $slaCheck.artifactPath
                expectedSchema = $slaCheck.expectedSchema
                present        = $true
                generatedAtUtc = $generatedAtUtc.ToString('o')
                ageHours       = $ageHours
                maxAgeHours    = $slaCheck.maxAgeHours
                withinSla      = $withinSla
            }) | Out-Null
    }
    catch {
        $violations.Add("SLA artifact timestamp parse failed: key=$($slaCheck.key) reason=$($_.Exception.Message)")
        $slaFreshness.Add([ordered]@{
                key            = $slaCheck.key
                artifactPath   = $slaCheck.artifactPath
                expectedSchema = $slaCheck.expectedSchema
                present        = $true
                maxAgeHours    = $slaCheck.maxAgeHours
                withinSla      = $false
            }) | Out-Null
    }
}

if ($planText -notmatch 'Validator tiering') {
    $violations.Add('Plan missing validator tiering section')
}

if ($planText -notmatch 'exhaustive fail SLA') {
    $violations.Add('Plan missing exhaustive fail SLA section')
}

$forbiddenRuntimeDependencyPatterns = @(
    @{ Pattern = '\.github/scripts/isr-verify-'; Message = 'runtime source must not depend on verifier scripts' },
    @{ Pattern = 'validator_tiering_report\.json'; Message = 'runtime source must not depend on validator tiering evidence artifact' },
    @{ Pattern = 'trigger_cleanup_completion_report\.json'; Message = 'runtime source must not depend on trigger cleanup completion evidence artifact' },
    @{ Pattern = '\bISR_REQUIRE_RUNTIME_EVIDENCE\b'; Message = 'runtime source must not depend on validator strict evidence environment flags' }
)

foreach ($sourceFile in $sourceFiles) {
    $content = Get-Content -LiteralPath $sourceFile.FullName -Raw -Encoding UTF8
    foreach ($rule in $forbiddenRuntimeDependencyPatterns) {
        if ($content -match $rule.Pattern) {
            $violations.Add("$($rule.Message): $($sourceFile.FullName)")
        }
    }
}

$report = [ordered]@{
    schema                 = 'validator_tiering_report_v3'
    generatedAt            = (Get-Date -Format 'o')
    tierRunnerPath         = $tierRunnerPath
    workflowPath           = $workflowPath
    planPath               = $planPath
    policyPath             = $policyPath
    policy                 = $policy
    slaFreshness           = $slaFreshness
    sourceRoot             = $sourceRoot
    scannedSourceFileCount = @($sourceFiles).Count
    violations             = $violations
}

Set-Content -LiteralPath $reportPath -Value ($report | ConvertTo-Json -Depth 6) -Encoding UTF8
Write-Host "[INFO] validator tiering report written: $reportPath"

if ($violations.Count -gt 0) {
    foreach ($violation in $violations) {
        Write-Host "[ERROR] $violation"
    }
    throw "Validator tiering violations detected. count=$($violations.Count)"
}

Write-Host '[PASS] validator tiering gate verified'
