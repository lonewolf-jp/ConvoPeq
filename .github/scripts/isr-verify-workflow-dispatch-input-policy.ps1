$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$workflowPath = Join-Path $repoRoot '.github\workflows\isr-verification.yml'
$policyPath = Join-Path $repoRoot '.github\isr-workflow-dispatch-input-policy.json'
$evidenceDir = Join-Path $repoRoot 'evidence'
$reportPath = Join-Path $evidenceDir 'workflow_dispatch_input_policy_report.json'

foreach ($path in @($workflowPath, $policyPath)) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Missing required file: $path"
    }
}

if (-not (Test-Path -LiteralPath $evidenceDir)) {
    New-Item -Path $evidenceDir -ItemType Directory | Out-Null
}

$workflowText = Get-Content -LiteralPath $workflowPath -Raw -Encoding UTF8
$policy = Get-Content -LiteralPath $policyPath -Raw -Encoding UTF8 | ConvertFrom-Json
$violations = New-Object System.Collections.Generic.List[string]

foreach ($field in @('schema', 'owner', 'issue', 'rationale', 'expiry', 'inputContract')) {
    if ($null -eq $policy.$field -or [string]::IsNullOrWhiteSpace("$($policy.$field)")) {
        $violations.Add("Workflow dispatch input policy missing required field: $field")
    }
}

if ($null -eq $policy.forwardingContract -or $null -eq $policy.forwardingContract.switches) {
    $violations.Add('Workflow dispatch input policy missing required field: forwardingContract.switches')
}

if ($null -eq $policy.forwardingContract -or $null -eq $policy.forwardingContract.arguments) {
    $violations.Add('Workflow dispatch input policy missing required field: forwardingContract.arguments')
}

if ("$($policy.schema)" -ne 'isr_workflow_dispatch_input_policy_v1') {
    $violations.Add("Workflow dispatch input policy schema mismatch: expected=isr_workflow_dispatch_input_policy_v1 actual=$($policy.schema)")
}

try {
    $expiry = [datetime]::ParseExact("$($policy.expiry)", 'yyyy-MM-dd', [System.Globalization.CultureInfo]::InvariantCulture)
    if ((Get-Date).Date -gt $expiry.Date) {
        $violations.Add("Workflow dispatch input policy expired: $($policy.expiry)")
    }
}
catch {
    $violations.Add("Workflow dispatch input policy expiry parse failed: '$($policy.expiry)' (expected yyyy-MM-dd)")
}

if ($null -eq $policy.inputContract.inputs) {
    $violations.Add('Workflow dispatch input policy missing required field: inputContract.inputs')
}

$contractInputs = @($policy.inputContract.inputs)
if ($contractInputs.Count -eq 0) {
    $violations.Add('Workflow dispatch input policy inputContract.inputs requires non-empty inputs')
}

$forwardingSwitches = @($policy.forwardingContract.switches)
if ($forwardingSwitches.Count -eq 0) {
    $violations.Add('Workflow dispatch input policy forwardingContract.switches requires non-empty switches')
}

$forwardingArguments = @($policy.forwardingContract.arguments)
if ($forwardingArguments.Count -eq 0) {
    $violations.Add('Workflow dispatch input policy forwardingContract.arguments requires non-empty arguments')
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

function Get-WorkflowInputProperty {
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

function Get-WorkflowInputOptions {
    param(
        [string]$Block
    )

    if ([string]::IsNullOrWhiteSpace($Block)) {
        return @()
    }

    $optionsMatch = [regex]::Match($Block, '(?ms)^\s{8}options:\s*\r?\n(?<options>(?:\s{10}-\s*[^\r\n]+\r?\n)+)')
    if (-not $optionsMatch.Success) {
        return @()
    }

    return @([regex]::Matches($optionsMatch.Groups['options'].Value, '(?m)^\s{10}-\s*(?<opt>[^\r\n]+)\s*$') | ForEach-Object { $_.Groups['opt'].Value.Trim() })
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

$contractInputNameSet = New-Object 'System.Collections.Generic.HashSet[string]'
$contractInputTypeMap = @{}
$inputStatus = New-Object System.Collections.Generic.List[object]

foreach ($entry in $contractInputs) {
    $inputName = "$($entry.name)"
    $expectedType = "$($entry.type)"
    $expectedRequired = "$($entry.required)".ToLowerInvariant()
    $expectedDefault = "$($entry.default)"
    $descriptionMustContain = "$($entry.descriptionMustContain)"

    if ([string]::IsNullOrWhiteSpace($inputName)) {
        $violations.Add('Workflow dispatch input policy has entry with empty name')
        continue
    }

    if (-not $contractInputNameSet.Add($inputName)) {
        $violations.Add("Workflow dispatch input policy has duplicate input name: $inputName")
    }
    else {
        $contractInputTypeMap[$inputName] = $expectedType
    }

    if ($expectedType -ne 'boolean' -and $expectedType -ne 'choice' -and $expectedType -ne 'string') {
        $violations.Add("Workflow dispatch input policy has invalid type: input=$inputName type=$expectedType")
    }

    if ($expectedType -eq 'boolean' -and $expectedDefault -ne 'true' -and $expectedDefault -ne 'false') {
        $violations.Add("Workflow dispatch input policy boolean input has invalid default: input=$inputName default=$expectedDefault")
    }

    if ($expectedType -eq 'string' -and $null -eq $entry.default) {
        $violations.Add("Workflow dispatch input policy string input has missing default value: input=$inputName")
    }

    if ($expectedRequired -ne 'true' -and $expectedRequired -ne 'false') {
        $violations.Add("Workflow dispatch input policy has invalid required value: input=$inputName required=$($entry.required)")
    }

    if ($expectedType -ne 'string' -and [string]::IsNullOrWhiteSpace($expectedDefault)) {
        $violations.Add("Workflow dispatch input policy has empty default: input=$inputName")
    }

    if ([string]::IsNullOrWhiteSpace($descriptionMustContain)) {
        $violations.Add("Workflow dispatch input policy missing descriptionMustContain: input=$inputName")
    }

    $block = Get-WorkflowInputBlock -WorkflowContent $workflowText -InputName $inputName
    $actualType = $null
    $actualRequired = $null
    $actualDefault = $null
    $actualDescription = $null
    $actualOptions = @()

    if ($null -eq $block) {
        $violations.Add("Workflow dispatch input mismatch: missing input block: $inputName")
    }
    else {
        $actualType = Get-WorkflowInputProperty -Block $block -PropertyName 'type'
        $actualRequired = Get-WorkflowInputProperty -Block $block -PropertyName 'required'
        $actualDefault = Get-WorkflowInputProperty -Block $block -PropertyName 'default'
        $actualDescription = Get-WorkflowInputProperty -Block $block -PropertyName 'description'
        $actualOptions = @(Get-WorkflowInputOptions -Block $block)

        if ($actualType -ne $expectedType) {
            $violations.Add("Workflow dispatch input mismatch: type: input=$inputName expected=$expectedType actual=$actualType")
        }

        if ($actualRequired -ne $expectedRequired) {
            $violations.Add("Workflow dispatch input mismatch: required: input=$inputName expected=$expectedRequired actual=$actualRequired")
        }

        if ($actualDefault -ne $expectedDefault) {
            $violations.Add("Workflow dispatch input mismatch: default: input=$inputName expected=$expectedDefault actual=$actualDefault")
        }

        if ([string]::IsNullOrWhiteSpace($actualDescription) -or -not $actualDescription.Contains($descriptionMustContain)) {
            $violations.Add("Workflow dispatch input mismatch: description: input=$inputName must include '$descriptionMustContain'")
        }

        if ($expectedType -eq 'choice') {
            $expectedOptions = @($entry.options)
            if ($expectedOptions.Count -eq 0) {
                $violations.Add("Workflow dispatch input policy choice input missing options: input=$inputName")
            }
            else {
                $optionSet = New-Object 'System.Collections.Generic.HashSet[string]'
                foreach ($option in $expectedOptions) {
                    $optionValue = "$option"
                    if ([string]::IsNullOrWhiteSpace($optionValue)) {
                        $violations.Add("Workflow dispatch input policy choice input has empty option: input=$inputName")
                        continue
                    }

                    if (-not $optionSet.Add($optionValue)) {
                        $violations.Add("Workflow dispatch input policy choice input has duplicate option: input=$inputName option=$optionValue")
                    }
                }

                if (-not $optionSet.Contains($expectedDefault)) {
                    $violations.Add("Workflow dispatch input policy choice input default is not in options: input=$inputName default=$expectedDefault")
                }
            }

            if ($actualOptions.Count -gt 0) {
                $actualOptionSet = New-Object 'System.Collections.Generic.HashSet[string]'
                foreach ($actualOption in $actualOptions) {
                    if (-not $actualOptionSet.Add("$actualOption")) {
                        $violations.Add("Workflow dispatch input mismatch: workflow has duplicate option: input=$inputName option=$actualOption")
                    }
                }
            }

            if (($actualOptions -join ',') -ne ($expectedOptions -join ',')) {
                $violations.Add("Workflow dispatch input mismatch: options: input=$inputName expected=$($expectedOptions -join '|') actual=$($actualOptions -join '|')")
            }
        }
    }

    $inputStatus.Add([ordered]@{
            name                   = $inputName
            expectedType           = $expectedType
            actualType             = $actualType
            expectedRequired       = $expectedRequired
            actualRequired         = $actualRequired
            expectedDefault        = $expectedDefault
            actualDefault          = $actualDefault
            expectedDescriptionKey = $descriptionMustContain
            actualDescription      = $actualDescription
            expectedOptions        = @($entry.options)
            actualOptions          = $actualOptions
        }) | Out-Null
}

$workflowInputNames = @(Get-WorkflowDispatchInputNames -WorkflowContent $workflowText)
foreach ($workflowInputName in $workflowInputNames) {
    if ($workflowInputName -notmatch '81' -and -not $contractInputNameSet.Contains($workflowInputName)) {
        $violations.Add("Workflow dispatch input mismatch: uncontracted non-8.1 workflow input detected: $workflowInputName")
    }
}

$forwardingStatus = New-Object System.Collections.Generic.List[object]
$forwardingInputNameSet = New-Object 'System.Collections.Generic.HashSet[string]'
$forwardingRunnerSwitchSet = New-Object 'System.Collections.Generic.HashSet[string]'
foreach ($switchContract in $forwardingSwitches) {
    $inputName = "$($switchContract.inputName)"
    $runnerSwitch = "$($switchContract.runnerSwitch)"

    if ([string]::IsNullOrWhiteSpace($inputName)) {
        $violations.Add('Workflow dispatch input policy forwardingContract has entry with empty inputName')
        continue
    }

    if ([string]::IsNullOrWhiteSpace($runnerSwitch)) {
        $violations.Add("Workflow dispatch input policy forwardingContract has empty runnerSwitch: input=$inputName")
        continue
    }

    if (-not $forwardingInputNameSet.Add($inputName)) {
        $violations.Add("Workflow dispatch input policy forwardingContract has duplicate inputName: input=$inputName")
    }

    if (-not $forwardingRunnerSwitchSet.Add($runnerSwitch)) {
        $violations.Add("Workflow dispatch input policy forwardingContract has duplicate runnerSwitch: switch=$runnerSwitch")
    }

    if (-not $contractInputNameSet.Contains($inputName)) {
        $violations.Add("Workflow dispatch input policy forwardingContract references unknown input: input=$inputName")
    }
    elseif ($contractInputTypeMap[$inputName] -ne 'boolean') {
        $violations.Add("Workflow dispatch input policy forwardingContract requires boolean input type: input=$inputName type=$($contractInputTypeMap[$inputName])")
    }

    $inputReferenceText = "inputs.$inputName"
    $runnerSwitchLiteral = "'$runnerSwitch'"
    $inputReferenced = $workflowText.Contains($inputReferenceText)
    $switchForwarded = $workflowText.Contains($runnerSwitchLiteral)

    if (-not $inputReferenced) {
        $violations.Add("Workflow dispatch forwarding mismatch: missing workflow input reference: input=$inputName text=$inputReferenceText")
    }

    if (-not $switchForwarded) {
        $violations.Add("Workflow dispatch forwarding mismatch: missing tier runner switch forwarding: input=$inputName switch=$runnerSwitch")
    }

    $forwardingStatus.Add([ordered]@{
            inputName       = $inputName
            forwardingKind  = 'switch'
            runnerSwitch    = $runnerSwitch
            inputReferenced = $inputReferenced
            switchForwarded = $switchForwarded
        }) | Out-Null
}

foreach ($argumentContract in $forwardingArguments) {
    $inputName = "$($argumentContract.inputName)"
    $runnerSwitch = "$($argumentContract.runnerSwitch)"
    $valueMode = "$($argumentContract.valueMode)"

    if ([string]::IsNullOrWhiteSpace($inputName)) {
        $violations.Add('Workflow dispatch input policy argument contract has entry with empty inputName')
        continue
    }

    if ([string]::IsNullOrWhiteSpace($runnerSwitch)) {
        $violations.Add("Workflow dispatch input policy argument contract has empty runnerSwitch: input=$inputName")
        continue
    }

    if ([string]::IsNullOrWhiteSpace($valueMode)) {
        $violations.Add("Workflow dispatch input policy argument contract has empty valueMode: input=$inputName")
        continue
    }

    if (-not $forwardingInputNameSet.Add($inputName)) {
        $violations.Add("Workflow dispatch input policy argument contract has duplicate inputName: input=$inputName")
    }

    if (-not $forwardingRunnerSwitchSet.Add($runnerSwitch)) {
        $violations.Add("Workflow dispatch input policy argument contract has duplicate runnerSwitch: switch=$runnerSwitch")
    }

    if (-not $contractInputNameSet.Contains($inputName)) {
        $violations.Add("Workflow dispatch input policy argument contract references unknown input: input=$inputName")
    }
    else {
        $inputType = $contractInputTypeMap[$inputName]
        if ($valueMode -eq 'rawNonEmpty') {
            if ($inputType -ne 'string' -and $inputType -ne 'choice') {
                $violations.Add("Workflow dispatch input policy argument contract requires string-or-choice input type: input=$inputName type=$inputType")
            }
        }
        elseif ($valueMode -eq 'nonNegativeInt') {
            if ($inputType -ne 'string') {
                $violations.Add("Workflow dispatch input policy argument contract requires string input type for nonNegativeInt: input=$inputName type=$inputType")
            }
        }
        else {
            $violations.Add("Workflow dispatch input policy argument contract has invalid valueMode: input=$inputName valueMode=$valueMode")
        }
    }

    $inputReferenceText = "inputs.$inputName"
    $inputReferenced = $workflowText.Contains($inputReferenceText)
    $switchForwarded = $workflowText.Contains("'$runnerSwitch'")

    if (-not $inputReferenced) {
        $violations.Add("Workflow dispatch argument mismatch: missing workflow input reference: input=$inputName text=$inputReferenceText")
    }

    if (-not $switchForwarded) {
        $violations.Add("Workflow dispatch argument mismatch: missing tier runner argument forwarding: input=$inputName switch=$runnerSwitch")
    }

    $forwardingStatus.Add([ordered]@{
            inputName       = $inputName
            forwardingKind  = 'argument'
            runnerSwitch    = $runnerSwitch
            valueMode       = $valueMode
            inputReferenced = $inputReferenced
            switchForwarded = $switchForwarded
        }) | Out-Null
}

$report = [ordered]@{
    schema           = 'workflow_dispatch_input_policy_report_v1'
    generatedAt      = (Get-Date -Format 'o')
    workflowPath     = $workflowPath
    policyPath       = $policyPath
    inputStatus      = $inputStatus
    forwardingStatus = $forwardingStatus
    violations       = $violations
}

Set-Content -LiteralPath $reportPath -Value ($report | ConvertTo-Json -Depth 8) -Encoding UTF8
Write-Host "[INFO] workflow dispatch input policy report written: $reportPath"

if ($violations.Count -gt 0) {
    foreach ($violation in $violations) {
        Write-Host "[ERROR] $violation"
    }
    throw "Workflow dispatch input policy violations detected. count=$($violations.Count)"
}

Write-Host '[PASS] workflow dispatch input policy gate verified'
