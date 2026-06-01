$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$contractPath = Join-Path $repoRoot 'doc\work10\contracts\runtime_semantic_reachability.md'
$testPath = Join-Path $repoRoot 'src\tests\RuntimeSemanticSchemaValidationTests.cpp'
$dispatchPath = Join-Path $repoRoot 'src\audioengine\AudioEngine.RebuildDispatch.cpp'
$builderPath = Join-Path $repoRoot 'src\audioengine\RuntimeBuilder.cpp'
$tierRunnerPath = Join-Path $repoRoot '.github\scripts\isr-run-tiered-verification.ps1'
$evidenceDir = Join-Path $repoRoot 'evidence'
$reportPath = Join-Path $evidenceDir 'semantic_reachability_report.json'

if (-not (Test-Path -LiteralPath $evidenceDir)) {
    New-Item -ItemType Directory -Path $evidenceDir -Force | Out-Null
}

$violations = New-Object 'System.Collections.Generic.List[string]'
function Add-Violation([string]$m) { $violations.Add($m) | Out-Null }

function Require-Path([string]$path, [string]$label) {
    if (-not (Test-Path -LiteralPath $path)) {
        Add-Violation "Missing ${label}: $path"
        return $false
    }
    return $true
}

function Measure-TokenCoverage {
    param(
        [Parameter(Mandatory = $true)] [string]$Text,
        [Parameter(Mandatory = $true)] [string[]]$Tokens,
        [Parameter(Mandatory = $true)] [string]$Label
    )

    $found = @()
    foreach ($token in $Tokens) {
        if ($Text -match [regex]::Escape($token)) {
            $found += $token
        }
        else {
            Add-Violation "$Label missing token: $token"
        }
    }

    return [ordered]@{
        label    = $Label
        required = $Tokens.Count
        found    = $found.Count
        coverage = if ($Tokens.Count -gt 0) { [math]::Round(($found.Count / $Tokens.Count) * 100.0, 2) } else { 0.0 }
        missing  = @($Tokens | Where-Object { $found -notcontains $_ })
    }
}

$hasContract = Require-Path -path $contractPath -label 'contract'
$hasTest = Require-Path -path $testPath -label 'test'
$hasDispatch = Require-Path -path $dispatchPath -label 'rebuild dispatch source'
$hasBuilder = Require-Path -path $builderPath -label 'runtime builder source'
$hasTierRunner = Require-Path -path $tierRunnerPath -label 'tier runner'

$contractCoverage = $null
$testCoverage = $null
$triggerCoverage = $null
$semanticCoverage = $null
$tierCoverage = $null

if ($hasContract) {
    $contractText = Get-Content -LiteralPath $contractPath -Raw -Encoding UTF8
    $contractCoverage = Measure-TokenCoverage -Text $contractText -Tokens @(
        'CrossfadeComplete',
        'RetireSettled',
        'PublicationStable',
        'reachable',
        'dead-end'
    ) -Label 'Reachability contract'
}

if ($hasTest) {
    $testText = Get-Content -LiteralPath $testPath -Raw -Encoding UTF8
    $testCoverage = Measure-TokenCoverage -Text $testText -Tokens @(
        'testRuntimeSemanticReachabilityValidation',
        'testSemanticTriggerToHashPathContract',
        'testSemanticHashCoverageContract',
        'testDescriptorCoverageContract'
    ) -Label 'Reachability unit test'
}

if ($hasDispatch) {
    $dispatchText = Get-Content -LiteralPath $dispatchPath -Raw -Encoding UTF8
    $triggerCoverage = Measure-TokenCoverage -Text $dispatchText -Tokens @(
        'submitRebuildIntent(',
        'requestRebuild(',
        'captureRuntimeBuildSnapshot(',
        'finalizeRuntimeBuildSnapshot(',
        'sealRuntimeBuildSnapshot(',
        'enqueuePublicationIntentForRuntimeCommit('
    ) -Label 'Trigger path coverage'
}

if ($hasBuilder) {
    $builderText = Get-Content -LiteralPath $builderPath -Raw -Encoding UTF8
    $semanticCoverage = Measure-TokenCoverage -Text $builderText -Tokens @(
        'worldOwner->semanticHash.generationSemanticHash',
        'worldOwner->semanticHash.topologyHash',
        'worldOwner->semanticHash.executionHash',
        'worldOwner->semanticHash.routingHash',
        'worldOwner->semanticHash.payloadHash',
        'worldOwner->semanticHash.publicationSemanticHash',
        'worldOwner->semanticHash.overlapSemanticHash',
        'worldOwner->semanticHash.retireSemanticHash'
    ) -Label 'Semantic hash coverage'
}

if ($hasTierRunner) {
    $tierText = Get-Content -LiteralPath $tierRunnerPath -Raw -Encoding UTF8
    $tierCoverage = Measure-TokenCoverage -Text $tierText -Tokens @(
        '.github/scripts/isr-verify-semantic-reachability.ps1'
    ) -Label 'Tier wiring coverage'
}

foreach ($coverage in @($contractCoverage, $testCoverage, $triggerCoverage, $semanticCoverage, $tierCoverage)) {
    if ($null -eq $coverage) {
        continue
    }

    if ($coverage.found -ne $coverage.required) {
        Add-Violation "$($coverage.label) is not fail-closed: found=$($coverage.found) required=$($coverage.required)"
    }
}

$report = [ordered]@{
    schema      = 'semantic_reachability_report_v2'
    generatedAt = (Get-Date -Format 'o')
    inputs      = [ordered]@{
        contractPath   = $contractPath
        testPath       = $testPath
        dispatchPath   = $dispatchPath
        builderPath    = $builderPath
        tierRunnerPath = $tierRunnerPath
    }
    coverage    = [ordered]@{
        contract     = $contractCoverage
        unitTest     = $testCoverage
        triggerPath  = $triggerCoverage
        semanticHash = $semanticCoverage
        tierWiring   = $tierCoverage
    }
    violations  = @($violations)
    ready       = ($violations.Count -eq 0)
}

$report | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $reportPath -Encoding UTF8
Write-Host "[INFO] report: $reportPath"

if ($violations.Count -gt 0) {
    foreach ($v in $violations) {
        Write-Host "[ERROR] $v"
    }
    throw "semantic reachability verification failed"
}

Write-Host '[PASS] semantic reachability verification passed'
