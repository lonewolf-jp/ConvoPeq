$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$workflowPath = Join-Path $repoRoot '.github\workflows\isr-verification.yml'
$policyPath = Join-Path $repoRoot '.github\isr-8_1-close-policy.json'
$evidenceDir = Join-Path $repoRoot 'evidence'
$reportPath = Join-Path $evidenceDir 'close_policy_8_1_workflow_input_contract_report.json'

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

function Get-WorkflowInputBlock {
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

    return $blockMatch.Groups['block'].Value
}

function Get-InputProperty {
    param(
        [string]$Block,
        [string]$PropertyName
    )

    if ([string]::IsNullOrWhiteSpace($Block)) {
        return $null
    }

    $pattern = "(?m)^\s{8}$([regex]::Escape($PropertyName)):\s*(?<value>[^\r\n]+)\s*$"
    $match = [regex]::Match($Block, $pattern)
    if (-not $match.Success) {
        return $null
    }

    $value = $match.Groups['value'].Value.Trim()
    if ($value.StartsWith('"') -and $value.EndsWith('"') -and $value.Length -ge 2) {
        return $value.Substring(1, $value.Length - 2)
    }

    return $value
}

function Resolve-PositiveInt {
    param(
        [string]$InputName,
        [string]$RawValue,
        [System.Collections.Generic.List[string]]$ViolationList
    )

    if ([string]::IsNullOrWhiteSpace($RawValue)) {
        $ViolationList.Add("Workflow input value missing: $InputName")
        return $null
    }

    $parsed = 0
    if (-not [int]::TryParse($RawValue, [ref]$parsed) -or $parsed -le 0) {
        $ViolationList.Add("Workflow input value invalid: $InputName value='$RawValue' (positive integer required)")
        return $null
    }

    return $parsed
}

function Get-WorkflowDispatchInputNames {
    param(
        [string]$WorkflowContent
    )

    $dispatchMatch = [regex]::Match($WorkflowContent, '(?ms)^\s{2}workflow_dispatch:\s*\r?\n(?<dispatch>(?:\s{4}.*\r?\n)+)')
    if (-not $dispatchMatch.Success) {
        return @()
    }

    $dispatchBlock = $dispatchMatch.Groups['dispatch'].Value
    $inputsMatch = [regex]::Match($dispatchBlock, '(?ms)^\s{4}inputs:\s*\r?\n(?<inputs>(?:\s{6}.*\r?\n)+)')
    if (-not $inputsMatch.Success) {
        return @()
    }

    $inputsBlock = $inputsMatch.Groups['inputs'].Value
    return @([regex]::Matches($inputsBlock, '(?m)^\s{6}(?<name>[A-Za-z0-9_]+):\s*$') | ForEach-Object { $_.Groups['name'].Value })
}

$workflowInputContract = $policy.workflowInputContract
$descriptionMustContain = $null
$contractEntries = @()

if ($null -ne $workflowInputContract) {
    if ([string]::IsNullOrWhiteSpace("$($workflowInputContract.descriptionMustContain)")) {
        $violations.Add('8.1 close policy workflowInputContract missing required field: descriptionMustContain')
    }
    else {
        $descriptionMustContain = "$($workflowInputContract.descriptionMustContain)"
    }

    if ($null -eq $workflowInputContract.inputs) {
        $violations.Add('8.1 close policy workflowInputContract missing required field: inputs')
    }
    else {
        $contractEntries = @($workflowInputContract.inputs)
        if ($contractEntries.Count -eq 0) {
            $violations.Add('8.1 close policy workflowInputContract requires non-empty inputs')
        }
    }
}

$nameSet = New-Object 'System.Collections.Generic.HashSet[string]'
$policyInputNames = New-Object 'System.Collections.Generic.HashSet[string]'

$inputStatus = New-Object System.Collections.Generic.List[object]
foreach ($entry in $contractEntries) {
    $inputName = "$($entry.name)"
    $expectedType = "$($entry.type)"
    $expectedRequired = "$($entry.required)".ToLowerInvariant()
    $expectedDefault = "$($entry.default)"
    $policyMinField = "$($entry.policyMinField)"
    $policyMaxField = "$($entry.policyMaxField)"

    if ([string]::IsNullOrWhiteSpace($inputName)) {
        $violations.Add('8.1 close policy workflowInputContract has entry with empty name')
        continue
    }

    if (-not $nameSet.Add($inputName)) {
        $violations.Add("8.1 close policy workflowInputContract has duplicate input name: $inputName")
    }

    [void]$policyInputNames.Add($inputName)

    if ($expectedType -ne 'boolean' -and $expectedType -ne 'string') {
        $violations.Add("8.1 close policy workflowInputContract input has invalid type: name=$inputName type=$expectedType")
    }

    if ($expectedRequired -ne 'true' -and $expectedRequired -ne 'false') {
        $violations.Add("8.1 close policy workflowInputContract input has invalid required: name=$inputName required=$($entry.required)")
    }

    if ([string]::IsNullOrWhiteSpace($expectedDefault)) {
        $violations.Add("8.1 close policy workflowInputContract input has empty default: name=$inputName")
    }

    $policyMin = $null
    $policyMax = $null
    if ($expectedType -eq 'string') {
        if ([string]::IsNullOrWhiteSpace($policyMinField) -or [string]::IsNullOrWhiteSpace($policyMaxField)) {
            $violations.Add("8.1 close policy workflowInputContract string input missing policy range fields: name=$inputName")
        }
        else {
            if ($null -eq $collectorPolicy.$policyMinField -or $null -eq $collectorPolicy.$policyMaxField) {
                $violations.Add("8.1 close policy workflowInputContract references unknown collector field: name=$inputName minField=$policyMinField maxField=$policyMaxField")
            }
            else {
                $policyMin = [int]$collectorPolicy.$policyMinField
                $policyMax = [int]$collectorPolicy.$policyMaxField
            }
        }
    }

    $block = Get-WorkflowInputBlock -WorkflowContent $workflowText -InputName $entry.name
    if ($null -eq $block) {
        $violations.Add("Workflow input block missing: $inputName")

        $inputStatus.Add([ordered]@{
                name              = $inputName
                blockPresent      = $false
                typeActual        = $null
                requiredActual    = $null
                defaultActual     = $null
                descriptionActual = $null
            }) | Out-Null
        continue
    }

    $typeActual = Get-InputProperty -Block $block -PropertyName 'type'
    $requiredActual = Get-InputProperty -Block $block -PropertyName 'required'
    $defaultActual = Get-InputProperty -Block $block -PropertyName 'default'
    $descriptionActual = Get-InputProperty -Block $block -PropertyName 'description'

    if ($typeActual -ne $expectedType) {
        $violations.Add("Workflow input type mismatch: $inputName expected=$expectedType actual=$typeActual")
    }

    if ($requiredActual -ne $expectedRequired) {
        $violations.Add("Workflow input required mismatch: $inputName expected=$expectedRequired actual=$requiredActual")
    }

    if ($defaultActual -ne $expectedDefault) {
        $violations.Add("Workflow input default mismatch: $inputName expected=$expectedDefault actual=$defaultActual")
    }

    if ([string]::IsNullOrWhiteSpace($descriptionActual) -or ($null -ne $descriptionMustContain -and -not $descriptionActual.Contains($descriptionMustContain))) {
        $violations.Add("Workflow input description contract mismatch: $inputName description must include '$descriptionMustContain'")
    }

    if ($expectedType -eq 'string' -and $null -ne $policyMin -and $null -ne $policyMax) {
        $parsedDefault = Resolve-PositiveInt -InputName $inputName -RawValue $defaultActual -ViolationList $violations
        if ($null -ne $parsedDefault) {
            if ($parsedDefault -lt $policyMin -or $parsedDefault -gt $policyMax) {
                $violations.Add("Workflow input default out of policy range: $inputName default=$parsedDefault allowed=$policyMin-$policyMax")
            }
        }
    }

    $inputStatus.Add([ordered]@{
            name              = $inputName
            blockPresent      = $true
            typeActual        = $typeActual
            requiredActual    = $requiredActual
            defaultActual     = $defaultActual
            descriptionActual = $descriptionActual
            expectedType      = $expectedType
            expectedRequired  = $expectedRequired
            expectedDefault   = $expectedDefault
            policyMinField    = $policyMinField
            policyMaxField    = $policyMaxField
            policyMin         = $policyMin
            policyMax         = $policyMax
        }) | Out-Null
}

$workflowInputNames = @(Get-WorkflowDispatchInputNames -WorkflowContent $workflowText)
foreach ($workflowInputName in $workflowInputNames) {
    if ($workflowInputName -match '81' -and -not $policyInputNames.Contains($workflowInputName)) {
        $violations.Add("Workflow input contract mismatch: uncontracted 8.1 workflow input detected: $workflowInputName")
    }
}

$report = [ordered]@{
    schema                 = 'close_policy_8_1_workflow_input_contract_report_v1'
    generatedAt            = (Get-Date -Format 'o')
    workflowPath           = $workflowPath
    policyPath             = $policyPath
    contractInputCount     = $contractEntries.Count
    descriptionMustContain = $descriptionMustContain
    inputStatus            = $inputStatus
    violations             = $violations
}

Set-Content -LiteralPath $reportPath -Value ($report | ConvertTo-Json -Depth 8) -Encoding UTF8
Write-Host "[INFO] 8.1 workflow input contract report written: $reportPath"

if ($violations.Count -gt 0) {
    foreach ($violation in $violations) {
        Write-Host "[ERROR] $violation"
    }

    throw "8.1 workflow input contract violations detected. count=$($violations.Count)"
}

Write-Host '[PASS] 8.1 workflow input contract gate verified'
