$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$evidenceDir = Join-Path $repoRoot 'evidence'
$reportPath = Join-Path $evidenceDir 'governance_consistency_report.json'

if (-not (Test-Path -LiteralPath $evidenceDir)) {
    New-Item -ItemType Directory -Path $evidenceDir -Force | Out-Null
}

$audioEngineHeaderPath = Join-Path $repoRoot 'src\audioengine\AudioEngine.h'
$semanticSchemaPath = Join-Path $repoRoot 'src\audioengine\ISRRuntimeSemanticSchema.h'
$verifierRegistryPath = Join-Path $repoRoot '.github\isr-verifier-registry.json'
$tieringPolicyPath = Join-Path $repoRoot '.github\isr-validator-tiering-policy.json'
$workflowPath = Join-Path $repoRoot '.github\workflows\isr-verification.yml'
$tierRunnerPath = Join-Path $repoRoot '.github\scripts\isr-run-tiered-verification.ps1'
$cmakeListsPath = Join-Path $repoRoot 'CMakeLists.txt'
$runtimeWorldAuthorityProjectionTestPath = Join-Path $repoRoot 'src\tests\RuntimeWorldAuthorityProjectionTests.cpp'
$crossfadeExecutorLocalContractTestPath = Join-Path $repoRoot 'src\tests\CrossfadeExecutorLocalContractTests.cpp'
$observePathSingleSourceTestPath = Join-Path $repoRoot 'src\tests\ObservePathSingleSourceTests.cpp'

$violations = New-Object 'System.Collections.Generic.List[string]'

function Add-Violation {
    param([string]$Message)
    $violations.Add($Message) | Out-Null
}

function Read-TextOrViolation {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Label
    )

    if (-not (Test-Path -LiteralPath $Path)) {
        Add-Violation "missing required file: $Label path=$Path"
        return $null
    }

    return Get-Content -LiteralPath $Path -Raw -Encoding UTF8
}

$audioHeader = Read-TextOrViolation -Path $audioEngineHeaderPath -Label 'audioEngineHeader'
$semanticSchema = Read-TextOrViolation -Path $semanticSchemaPath -Label 'runtimeSemanticSchema'
$workflowText = Read-TextOrViolation -Path $workflowPath -Label 'workflow'
$tierRunnerText = Read-TextOrViolation -Path $tierRunnerPath -Label 'tierRunner'
$cmakeListsText = Read-TextOrViolation -Path $cmakeListsPath -Label 'cmakeLists'

if ($null -ne $audioHeader) {
    if (-not $audioHeader.Contains('kFieldDescriptors')) {
        Add-Violation 'descriptor table mismatch: AudioEngine.h missing kFieldDescriptors'
    }

    if (-not $audioHeader.Contains('kRuntimeAuthorityInventory')) {
        Add-Violation 'authority inventory mismatch: AudioEngine.h missing kRuntimeAuthorityInventory'
    }

    if (-not $audioHeader.Contains('kRuntimeReadAuthorityInventory')) {
        Add-Violation 'read authority inventory mismatch: AudioEngine.h missing kRuntimeReadAuthorityInventory'
    }

    if (-not $audioHeader.Contains('validateAuthorityInventoryAgainstDescriptors(kRuntimeAuthorityInventory, kFieldDescriptors)')) {
        Add-Violation 'authority inventory mismatch: validateAuthorityInventoryAgainstDescriptors wiring missing'
    }

    if (-not $audioHeader.Contains('validateReadAuthorityInventoryAgainstDescriptors(kRuntimeReadAuthorityInventory, kFieldDescriptors)')) {
        Add-Violation 'read authority inventory mismatch: validateReadAuthorityInventoryAgainstDescriptors wiring missing'
    }

    if (-not $audioHeader.Contains('validateFieldDescriptorSet(kFieldDescriptors)')) {
        Add-Violation 'descriptor table mismatch: validateFieldDescriptorSet(kFieldDescriptors) wiring missing'
    }
}

if ($null -ne $semanticSchema) {
    if (-not $semanticSchema.Contains('validateFieldDescriptorSet')) {
        Add-Violation 'descriptor table mismatch: ISRRuntimeSemanticSchema.h missing validateFieldDescriptorSet'
    }

    if (-not $semanticSchema.Contains('validateAuthorityInventoryAgainstDescriptors')) {
        Add-Violation 'authority inventory mismatch: ISRRuntimeSemanticSchema.h missing validateAuthorityInventoryAgainstDescriptors'
    }

    if (-not $semanticSchema.Contains('validateReadAuthorityInventoryAgainstDescriptors')) {
        Add-Violation 'read authority inventory mismatch: ISRRuntimeSemanticSchema.h missing validateReadAuthorityInventoryAgainstDescriptors'
    }
}

if (Test-Path -LiteralPath $verifierRegistryPath) {
    $verifierRegistry = Get-Content -LiteralPath $verifierRegistryPath -Raw -Encoding UTF8 | ConvertFrom-Json

    if ("$($verifierRegistry.schema)" -ne 'isr_verifier_registry_v1') {
        Add-Violation "registry missing entry: schema mismatch actual=$($verifierRegistry.schema)"
    }

    $releaseTier = @($verifierRegistry.tiers.release)
    if ($releaseTier -notcontains 'governance-consistency-verifier') {
        Add-Violation 'registry missing entry: governance-consistency-verifier is absent from release tier'
    }

    $allVerifiers = New-Object 'System.Collections.Generic.HashSet[string]'
    foreach ($tierName in @('pr', 'nightly', 'release')) {
        foreach ($name in @($verifierRegistry.tiers.$tierName)) {
            $verifierName = "$name"
            if (-not [string]::IsNullOrWhiteSpace($verifierName)) {
                [void]$allVerifiers.Add($verifierName)
            }
        }
    }

    foreach ($dep in @($verifierRegistry.dependencies)) {
        $depVerifier = "$($dep.verifier)"
        if (-not [string]::IsNullOrWhiteSpace($depVerifier) -and -not $allVerifiers.Contains($depVerifier)) {
            Add-Violation "orphan verifier: dependency source is not present in tier list verifier=$depVerifier"
        }

        foreach ($depName in @($dep.dependsOn)) {
            $target = "$depName"
            if (-not [string]::IsNullOrWhiteSpace($target) -and -not $allVerifiers.Contains($target)) {
                Add-Violation "orphan verifier: dependency target is not present in tier list verifier=$depVerifier dependsOn=$target"
            }
        }
    }
}
else {
    Add-Violation "registry missing entry: missing verifier registry file path=$verifierRegistryPath"
}

if (Test-Path -LiteralPath $tieringPolicyPath) {
    $tierPolicy = Get-Content -LiteralPath $tieringPolicyPath -Raw -Encoding UTF8 | ConvertFrom-Json
    if ("$($tierPolicy.schema)" -ne 'isr_validator_tiering_policy_v1') {
        Add-Violation "tier missing entry: tiering policy schema mismatch actual=$($tierPolicy.schema)"
    }

    foreach ($tierName in @('smoke', 'standard', 'exhaustive')) {
        if ($null -eq $tierPolicy.tiers.$tierName -or [string]::IsNullOrWhiteSpace("$($tierPolicy.tiers.$tierName)")) {
            Add-Violation "tier missing entry: tiers.$tierName is missing"
        }
    }

    if ($null -eq $tierPolicy.workflowSchedule -or [string]::IsNullOrWhiteSpace("$($tierPolicy.workflowSchedule.nightlyCron)") -or [string]::IsNullOrWhiteSpace("$($tierPolicy.workflowSchedule.weeklyCron)")) {
        Add-Violation 'tier missing entry: workflowSchedule nightly/weekly cron is missing'
    }
}
else {
    Add-Violation "tier missing entry: missing tiering policy file path=$tieringPolicyPath"
}

if ($null -ne $workflowText) {
    if (-not $workflowText.Contains('isr-run-tiered-verification.ps1')) {
        Add-Violation 'workflow missing registration: isr-run-tiered-verification.ps1 invocation is absent'
    }
}

if (-not (Test-Path -LiteralPath $runtimeWorldAuthorityProjectionTestPath)) {
    Add-Violation "contract test missing: RuntimeWorldAuthorityProjectionTests.cpp path=$runtimeWorldAuthorityProjectionTestPath"
}

if (-not (Test-Path -LiteralPath $crossfadeExecutorLocalContractTestPath)) {
    Add-Violation "contract test missing: CrossfadeExecutorLocalContractTests.cpp path=$crossfadeExecutorLocalContractTestPath"
}

if (-not (Test-Path -LiteralPath $observePathSingleSourceTestPath)) {
    Add-Violation "contract test missing: ObservePathSingleSourceTests.cpp path=$observePathSingleSourceTestPath"
}

if ($null -ne $cmakeListsText) {
    if (-not $cmakeListsText.Contains('add_executable(RuntimeWorldAuthorityProjectionTests')) {
        Add-Violation 'cmake wiring mismatch: RuntimeWorldAuthorityProjectionTests executable registration is missing'
    }

    if (-not $cmakeListsText.Contains('add_test(NAME RuntimeWorldAuthorityProjectionContract COMMAND RuntimeWorldAuthorityProjectionTests)')) {
        Add-Violation 'cmake wiring mismatch: RuntimeWorldAuthorityProjectionContract add_test wiring is missing'
    }

    if (-not $cmakeListsText.Contains('add_executable(CrossfadeExecutorLocalContractTests')) {
        Add-Violation 'cmake wiring mismatch: CrossfadeExecutorLocalContractTests executable registration is missing'
    }

    if (-not $cmakeListsText.Contains('add_test(NAME CrossfadeExecutorLocalContract COMMAND CrossfadeExecutorLocalContractTests)')) {
        Add-Violation 'cmake wiring mismatch: CrossfadeExecutorLocalContract add_test wiring is missing'
    }

    if (-not $cmakeListsText.Contains('add_executable(ObservePathSingleSourceTests')) {
        Add-Violation 'cmake wiring mismatch: ObservePathSingleSourceTests executable registration is missing'
    }

    if (-not $cmakeListsText.Contains('add_test(NAME ObservePathSingleSource COMMAND ObservePathSingleSourceTests)')) {
        Add-Violation 'cmake wiring mismatch: ObservePathSingleSource add_test wiring is missing'
    }
}

if ($null -ne $tierRunnerText) {
    if (-not $tierRunnerText.Contains("'.github/scripts/isr-verify-governance-consistency.ps1'")) {
        Add-Violation 'workflow missing registration: governance consistency verifier script is not wired in tier runner'
    }
}

if ($null -ne $audioHeader) {
    if ($audioHeader.Contains('kRuntimeAuthorityInventory') -and -not $audioHeader.Contains('kAuthorityInventory = kRuntimeAuthorityInventory')) {
        Add-Violation 'orphan inventory: kRuntimeAuthorityInventory is not canonicalized via kAuthorityInventory alias'
    }

    if ($audioHeader.Contains('kFieldDescriptors') -and -not $audioHeader.Contains('validateDescriptorSet()')) {
        Add-Violation 'orphan descriptor: kFieldDescriptors exists but validateDescriptorSet() contract is missing'
    }
}

$report = [ordered]@{
    schema      = 'governance_consistency_report_v1'
    generatedAt = (Get-Date -Format 'o')
    violations  = @($violations)
    ready       = ($violations.Count -eq 0)
}

$report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $reportPath -Encoding UTF8
Write-Host "[INFO] governance consistency report written: $reportPath"

if ($violations.Count -gt 0) {
    foreach ($violation in $violations) {
        Write-Host "[ERROR] $violation"
    }

    throw "Governance consistency verification failed. violations=$($violations.Count)"
}

Write-Host '[PASS] governance consistency verification passed'
