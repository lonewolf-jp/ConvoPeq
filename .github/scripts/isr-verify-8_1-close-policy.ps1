param(
    [ValidateSet('smoke', 'standard', 'exhaustive')]
    [string]$Tier = 'standard'
)

$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$policyPath = Join-Path $repoRoot '.github\isr-8_1-close-policy.json'
$evidenceDir = Join-Path $repoRoot 'evidence'
$reportPath = Join-Path $evidenceDir 'close_policy_8_1_report.json'

if (-not (Test-Path -LiteralPath $policyPath)) {
    throw "Missing 8.1 close policy: $policyPath"
}

if (-not (Test-Path -LiteralPath $evidenceDir)) {
    New-Item -Path $evidenceDir -ItemType Directory | Out-Null
}

$policy = Get-Content -LiteralPath $policyPath -Raw -Encoding UTF8 | ConvertFrom-Json
$violations = New-Object System.Collections.Generic.List[string]

if ("$($policy.schema)" -ne 'isr_8_1_close_policy_v1') {
    $violations.Add("Unexpected 8.1 close policy schema: $($policy.schema)")
}

foreach ($field in @('owner', 'issue', 'rationale', 'expiry', 'collector')) {
    if ($null -eq $policy.$field -or [string]::IsNullOrWhiteSpace("$($policy.$field)")) {
        $violations.Add("8.1 close policy missing required field: $field")
    }
}

if ($null -eq $policy.expiryGuardDaysByTier) {
    $violations.Add('8.1 close policy missing required field: expiryGuardDaysByTier')
}

$expiryValid = $false
$expired = $false
if (-not [string]::IsNullOrWhiteSpace("$($policy.expiry)")) {
    try {
        $expiry = [datetime]::ParseExact("$($policy.expiry)", 'yyyy-MM-dd', [System.Globalization.CultureInfo]::InvariantCulture)
        $expiryValid = $true
        $expired = (Get-Date).Date -gt $expiry.Date
        if ($expired) {
            $violations.Add("8.1 close policy expired: expiry=$($policy.expiry)")
        }
    }
    catch {
        $violations.Add("8.1 close policy has invalid expiry format: '$($policy.expiry)' (expected yyyy-MM-dd)")
    }
}

$expiryGuardDays = $null
if ($null -ne $policy.expiryGuardDaysByTier) {
    if ($null -eq $policy.expiryGuardDaysByTier.standard -or $null -eq $policy.expiryGuardDaysByTier.exhaustive) {
        $violations.Add('8.1 close policy expiryGuardDaysByTier requires standard/exhaustive fields')
    }
    else {
        if ([int]$policy.expiryGuardDaysByTier.standard -lt 1 -or [int]$policy.expiryGuardDaysByTier.exhaustive -lt 1) {
            $violations.Add('8.1 close policy expiryGuardDaysByTier values must be >= 1')
        }
        elseif ([int]$policy.expiryGuardDaysByTier.standard -gt [int]$policy.expiryGuardDaysByTier.exhaustive) {
            $violations.Add('8.1 close policy expiryGuardDaysByTier invalid: standard must be <= exhaustive')
        }
        elseif ($Tier -eq 'standard' -or $Tier -eq 'exhaustive') {
            $expiryGuardDays = [int]$policy.expiryGuardDaysByTier.$Tier
        }
    }
}

$expiryDaysRemaining = $null
if ($expiryValid) {
    $expiryDaysRemaining = [int][math]::Floor(($expiry.Date - (Get-Date).Date).TotalDays)
    if ($Tier -eq 'standard' -or $Tier -eq 'exhaustive') {
        if ($null -eq $expiryGuardDays) {
            $violations.Add("8.1 close policy missing expiry guard for tier=$Tier")
        }
        elseif ($expiryDaysRemaining -lt $expiryGuardDays) {
            $violations.Add("8.1 close policy expiry guard breached for tier=${Tier}: daysRemaining=$expiryDaysRemaining guardDays=$expiryGuardDays")
        }
    }
}

$collector = $policy.collector
$numericFields = @(
    'minWindowSec',
    'maxWindowSec',
    'minAutoCaptureTimeoutSec',
    'maxAutoCaptureTimeoutSec',
    'minProbeExitMs',
    'maxProbeExitMs',
    'minRetryMax',
    'maxRetryMax'
)

if ($null -ne $collector) {
    foreach ($field in $numericFields) {
        if ($null -eq $collector.$field) {
            $violations.Add("8.1 close policy collector missing required field: $field")
        }
        elseif ([int]$collector.$field -lt 1) {
            $violations.Add("8.1 close policy collector requires $field >= 1 but was $($collector.$field)")
        }
    }

    $rangePairs = @(
        @{ Min = 'minWindowSec'; Max = 'maxWindowSec' },
        @{ Min = 'minAutoCaptureTimeoutSec'; Max = 'maxAutoCaptureTimeoutSec' },
        @{ Min = 'minProbeExitMs'; Max = 'maxProbeExitMs' },
        @{ Min = 'minRetryMax'; Max = 'maxRetryMax' }
    )

    foreach ($pair in $rangePairs) {
        if ($null -ne $collector.($pair.Min) -and $null -ne $collector.($pair.Max)) {
            if ([int]$collector.($pair.Min) -gt [int]$collector.($pair.Max)) {
                $violations.Add("8.1 close policy collector invalid range: $($pair.Min) > $($pair.Max)")
            }
        }
    }

    foreach ($tierField in @('allowedCollectTiers', 'allowedEnforceTiers')) {
        if ($null -eq $collector.$tierField) {
            $violations.Add("8.1 close policy collector missing required field: $tierField")
            continue
        }

        $tiers = @($collector.$tierField)
        if ($tiers.Count -eq 0) {
            $violations.Add("8.1 close policy collector requires non-empty $tierField")
            continue
        }

        foreach ($tier in $tiers) {
            if (@('standard', 'exhaustive') -notcontains "$tier") {
                $violations.Add("8.1 close policy collector has invalid $tierField tier: $tier")
            }
        }
    }

    if (@($collector.allowedCollectTiers) -notcontains 'standard') {
        $violations.Add('8.1 close policy collector must allow standard tier for collect path')
    }

    if (@($collector.allowedEnforceTiers) -notcontains 'standard') {
        $violations.Add('8.1 close policy collector must allow standard tier for enforce path')
    }
}

$workflowInputContract = $policy.workflowInputContract
if ($null -eq $workflowInputContract) {
    $violations.Add('8.1 close policy missing workflowInputContract section')
}
else {
    if ([string]::IsNullOrWhiteSpace("$($workflowInputContract.descriptionMustContain)")) {
        $violations.Add('8.1 close policy workflowInputContract missing required field: descriptionMustContain')
    }

    if ($null -eq $workflowInputContract.inputs) {
        $violations.Add('8.1 close policy workflowInputContract missing required field: inputs')
    }
    else {
        $inputs = @($workflowInputContract.inputs)
        if ($inputs.Count -eq 0) {
            $violations.Add('8.1 close policy workflowInputContract requires non-empty inputs')
        }

        $seenInputNames = New-Object 'System.Collections.Generic.HashSet[string]'
        foreach ($inputContract in $inputs) {
            $inputName = "$($inputContract.name)"
            $inputType = "$($inputContract.type)"
            $inputRequired = "$($inputContract.required)".ToLowerInvariant()
            $inputDefault = "$($inputContract.default)"
            $policyMinField = "$($inputContract.policyMinField)"
            $policyMaxField = "$($inputContract.policyMaxField)"

            if ([string]::IsNullOrWhiteSpace($inputName)) {
                $violations.Add('8.1 close policy workflowInputContract has entry with empty name')
                continue
            }

            if (-not $seenInputNames.Add($inputName)) {
                $violations.Add("8.1 close policy workflowInputContract has duplicate input name: $inputName")
            }

            if ($inputType -ne 'boolean' -and $inputType -ne 'string') {
                $violations.Add("8.1 close policy workflowInputContract input has invalid type: name=$inputName type=$inputType")
            }

            if ($inputRequired -ne 'true' -and $inputRequired -ne 'false') {
                $violations.Add("8.1 close policy workflowInputContract input has invalid required: name=$inputName required=$($inputContract.required)")
            }

            if ([string]::IsNullOrWhiteSpace($inputDefault)) {
                $violations.Add("8.1 close policy workflowInputContract input has empty default: name=$inputName")
            }

            if ($inputType -eq 'string') {
                if ([string]::IsNullOrWhiteSpace($policyMinField) -or [string]::IsNullOrWhiteSpace($policyMaxField)) {
                    $violations.Add("8.1 close policy workflowInputContract string input missing policy range fields: name=$inputName")
                }
                elseif ($null -eq $collector.$policyMinField -or $null -eq $collector.$policyMaxField) {
                    $violations.Add("8.1 close policy workflowInputContract references unknown collector field: name=$inputName minField=$policyMinField maxField=$policyMaxField")
                }
            }
            else {
                if (-not [string]::IsNullOrWhiteSpace($policyMinField) -or -not [string]::IsNullOrWhiteSpace($policyMaxField)) {
                    $violations.Add("8.1 close policy workflowInputContract boolean input must not declare policy range fields: name=$inputName")
                }
            }
        }
    }
}

$report = [ordered]@{
    schema      = 'close_policy_8_1_report_v1'
    generatedAt = (Get-Date -Format 'o')
    policyPath  = $policyPath
    policy      = [ordered]@{
        schema                = "$($policy.schema)"
        owner                 = "$($policy.owner)"
        issue                 = "$($policy.issue)"
        rationale             = "$($policy.rationale)"
        expiry                = "$($policy.expiry)"
        expiryDaysRemaining   = $expiryDaysRemaining
        expiryValid           = $expiryValid
        expired               = $expired
        expiryGuardDaysByTier = $policy.expiryGuardDaysByTier
        activeTier            = $Tier
        activeTierGuardDays   = $expiryGuardDays
        collector             = $collector
    }
    violations  = $violations
}

Set-Content -LiteralPath $reportPath -Value ($report | ConvertTo-Json -Depth 8) -Encoding UTF8
Write-Host "[INFO] 8.1 close policy report written: $reportPath"

if ($violations.Count -gt 0) {
    foreach ($violation in $violations) {
        Write-Host "[ERROR] $violation"
    }
    throw "8.1 close policy violations detected. count=$($violations.Count)"
}

Write-Host '[PASS] 8.1 close policy gate verified'
