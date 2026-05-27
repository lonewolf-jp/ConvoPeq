$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$workflowPath = Join-Path $repoRoot '.github\workflows\isr-verification.yml'
$policyPath = Join-Path $repoRoot '.github\isr-8_1-close-policy.json'
$evidenceDir = Join-Path $repoRoot 'evidence'
$reportPath = Join-Path $evidenceDir 'close_policy_8_1_workflow_input_coherence_report.json'

if (-not (Test-Path -LiteralPath $workflowPath)) {
    throw "Missing workflow file: $workflowPath"
}

if (-not (Test-Path -LiteralPath $policyPath)) {
    throw "Missing 8.1 close policy: $policyPath"
}

if (-not (Test-Path -LiteralPath $evidenceDir)) {
    New-Item -Path $evidenceDir -ItemType Directory | Out-Null
}

$workflowText = Get-Content -LiteralPath $workflowPath -Raw -Encoding UTF8
$policy = Get-Content -LiteralPath $policyPath -Raw -Encoding UTF8 | ConvertFrom-Json
$violations = New-Object System.Collections.Generic.List[string]

if ("$($policy.schema)" -ne 'isr_8_1_close_policy_v1') {
    $violations.Add("Unexpected 8.1 close policy schema: $($policy.schema)")
}

if ($null -eq $policy.collector) {
    $violations.Add('8.1 close policy missing collector section')
}

if ($null -eq $policy.workflowInputContract) {
    $violations.Add('8.1 close policy missing workflowInputContract section')
}

$collectorPolicy = $policy.collector
foreach ($field in @('minWindowSec', 'maxWindowSec', 'minAutoCaptureTimeoutSec', 'maxAutoCaptureTimeoutSec', 'minProbeExitMs', 'maxProbeExitMs', 'minRetryMax', 'maxRetryMax')) {
    if ($null -eq $collectorPolicy.$field) {
        $violations.Add("8.1 close policy collector missing required field: $field")
    }
}

function Get-WorkflowInputDefault {
    param(
        [string]$WorkflowContent,
        [string]$InputName
    )

    $escapedName = [regex]::Escape($InputName)
    $blockPattern = "(?ms)^\s{6}${escapedName}:\s*\r?\n(?<block>(?:\s{8}.*\r?\n)+)"
    $blockMatch = [regex]::Match($WorkflowContent, $blockPattern)
    if (-not $blockMatch.Success) {
        return $null
    }

    $blockText = $blockMatch.Groups['block'].Value
    $defaultMatch = [regex]::Match($blockText, '(?m)^\s{8}default:\s*"?(?<value>[^"\r\n]+)"?\s*$')
    if (-not $defaultMatch.Success) {
        return $null
    }

    return $defaultMatch.Groups['value'].Value
}

function Resolve-PositiveIntDefault {
    param(
        [string]$InputName,
        [string]$RawValue,
        [System.Collections.Generic.List[string]]$ViolationList
    )

    if ([string]::IsNullOrWhiteSpace($RawValue)) {
        $ViolationList.Add("Workflow input default missing: $InputName")
        return $null
    }

    $parsed = 0
    if (-not [int]::TryParse($RawValue, [ref]$parsed) -or $parsed -le 0) {
        $ViolationList.Add("Workflow input default invalid: $InputName default='$RawValue' (positive integer required)")
        return $null
    }

    return $parsed
}

$defaults = [ordered]@{}
$resolvedDefaults = [ordered]@{}
$coherenceInputs = @()

if ($null -ne $policy.workflowInputContract -and $null -ne $policy.workflowInputContract.inputs) {
    $coherenceInputs = @($policy.workflowInputContract.inputs | Where-Object {
            "$($_.type)" -eq 'string' -and
            -not [string]::IsNullOrWhiteSpace("$($_.policyMinField)") -and
            -not [string]::IsNullOrWhiteSpace("$($_.policyMaxField)")
        })
}

if ($coherenceInputs.Count -eq 0) {
    $violations.Add('8.1 close policy workflowInputContract has no string inputs with policy range fields for coherence checks')
}

foreach ($entry in $coherenceInputs) {
    $inputName = "$($entry.name)"
    $policyMinField = "$($entry.policyMinField)"
    $policyMaxField = "$($entry.policyMaxField)"

    if ([string]::IsNullOrWhiteSpace($inputName)) {
        $violations.Add('8.1 close policy workflowInputContract has string range entry with empty name')
        continue
    }

    if ($null -eq $collectorPolicy.$policyMinField -or $null -eq $collectorPolicy.$policyMaxField) {
        $violations.Add("8.1 close policy workflowInputContract coherence references unknown collector field: name=$inputName minField=$policyMinField maxField=$policyMaxField")
        continue
    }

    $rawDefault = Get-WorkflowInputDefault -WorkflowContent $workflowText -InputName $inputName
    $defaults[$inputName] = $rawDefault

    $resolvedDefault = Resolve-PositiveIntDefault -InputName $inputName -RawValue $rawDefault -ViolationList $violations
    $resolvedDefaults[$inputName] = $resolvedDefault

    if ($null -ne $resolvedDefault) {
        $policyMin = [int]$collectorPolicy.$policyMinField
        $policyMax = [int]$collectorPolicy.$policyMaxField
        if ($resolvedDefault -lt $policyMin -or $resolvedDefault -gt $policyMax) {
            $violations.Add("Workflow input default out of policy range: $inputName default=$resolvedDefault allowed=$policyMin-$policyMax")
        }
    }
}

$report = [ordered]@{
    schema          = 'close_policy_8_1_workflow_input_coherence_report_v1'
    generatedAt     = (Get-Date -Format 'o')
    workflowPath    = $workflowPath
    policyPath      = $policyPath
    defaults        = $defaults
    resolvedDefaults = $resolvedDefaults
    collectorPolicy = $collectorPolicy
    violations      = $violations
}

Set-Content -LiteralPath $reportPath -Value ($report | ConvertTo-Json -Depth 8) -Encoding UTF8
Write-Host "[INFO] 8.1 workflow input coherence report written: $reportPath"

if ($violations.Count -gt 0) {
    foreach ($violation in $violations) {
        Write-Host "[ERROR] $violation"
    }
    throw "8.1 workflow input coherence violations detected. count=$($violations.Count)"
}

Write-Host '[PASS] 8.1 workflow input coherence gate verified'
