$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$evidenceDir = Join-Path $repoRoot 'evidence'
$reportPath = Join-Path $evidenceDir 'shadow_compare_contract_report.json'

if (-not (Test-Path -LiteralPath $evidenceDir)) {
    New-Item -ItemType Directory -Path $evidenceDir -Force | Out-Null
}

$contractRegistryPath = Join-Path $repoRoot '.github\isr-contract-registry.json'
$schemaHeaderPath = Join-Path $repoRoot 'src\audioengine\ISRRuntimeSemanticSchema.h'
$commitPath = Join-Path $repoRoot 'src\audioengine\AudioEngine.Commit.cpp'
$debugRuntimePath = Join-Path $repoRoot 'src\audioengine\ISRDebugRuntime.cpp'

$violations = New-Object 'System.Collections.Generic.List[string]'

function Add-Violation {
    param([string]$Message)
    $violations.Add($Message) | Out-Null
}

if (-not (Test-Path -LiteralPath $contractRegistryPath)) {
    Add-Violation "Missing contract registry: $contractRegistryPath"
}

if (-not (Test-Path -LiteralPath $schemaHeaderPath)) {
    Add-Violation "Missing schema header: $schemaHeaderPath"
}

if (-not (Test-Path -LiteralPath $commitPath)) {
    Add-Violation "Missing source file: $commitPath"
}

if (-not (Test-Path -LiteralPath $debugRuntimePath)) {
    Add-Violation "Missing source file: $debugRuntimePath"
}

if (Test-Path -LiteralPath $contractRegistryPath) {
    $contractRegistry = Get-Content -LiteralPath $contractRegistryPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $contractIds = @($contractRegistry.contracts | ForEach-Object { [string]$_.id })

    if ($contractIds -notcontains 'hash-authority-prohibition') {
        Add-Violation 'Required contract id missing: hash-authority-prohibition'
    }

    if ($contractIds -notcontains 'shadow-compare-cadence-contract') {
        Add-Violation 'Required contract id missing: shadow-compare-cadence-contract'
    }
}

if (Test-Path -LiteralPath $commitPath) {
    $commitText = Get-Content -LiteralPath $commitPath -Raw -Encoding UTF8
    if ($commitText -notmatch 'recordShadowCompareObservation\s*\(') {
        Add-Violation 'Shadow compare contract violation: commit path must record shadow compare observations.'
    }
}

if (Test-Path -LiteralPath $schemaHeaderPath) {
    $schemaText = Get-Content -LiteralPath $schemaHeaderPath -Raw -Encoding UTF8
    if ($schemaText -notmatch 'generationSemanticHash') {
        Add-Violation 'Shadow compare contract violation: RuntimeSemanticHash must include generationSemanticHash coverage.'
    }
}

if (Test-Path -LiteralPath $debugRuntimePath) {
    $debugText = Get-Content -LiteralPath $debugRuntimePath -Raw -Encoding UTF8

    if ($debugText -notmatch 'semanticHashEquals\s*\(') {
        Add-Violation 'Shadow compare contract violation: semantic hash equality helper is missing.'
    }

    if ($debugText -notmatch 'emitShadowCompareCadenceReport\s*\(') {
        Add-Violation 'Shadow compare contract violation: cadence report emission is missing.'
    }

    if ($debugText -notmatch 'lhs\.generationSemanticHash\s*==\s*rhs\.generationSemanticHash') {
        Add-Violation 'Shadow compare contract violation: semantic hash equality must include generationSemanticHash.'
    }
}

$report = [ordered]@{
    schema               = 'shadow_compare_contract_report_v1'
    generatedAt          = (Get-Date -Format 'o')
    contractRegistryPath = $contractRegistryPath
    schemaHeaderPath     = $schemaHeaderPath
    commitPath           = $commitPath
    debugRuntimePath     = $debugRuntimePath
    violations           = @($violations)
    ready                = ($violations.Count -eq 0)
}

$report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $reportPath -Encoding UTF8
Write-Host "[INFO] shadow compare contract report written: $reportPath"

if ($violations.Count -gt 0) {
    foreach ($v in $violations) {
        Write-Host "[ERROR] $v"
    }
    throw "Shadow compare contract verification failed. violations=$($violations.Count)"
}

Write-Host '[PASS] shadow compare contract verification passed'
