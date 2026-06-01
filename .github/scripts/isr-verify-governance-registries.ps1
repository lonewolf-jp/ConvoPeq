$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$evidenceDir = Join-Path $repoRoot 'evidence'
$reportPath = Join-Path $evidenceDir 'governance_registries_report.json'

if (-not (Test-Path -LiteralPath $evidenceDir)) {
    New-Item -ItemType Directory -Path $evidenceDir -Force | Out-Null
}

$contractRegistryPath = Join-Path $repoRoot '.github\isr-contract-registry.json'
$verifierRegistryPath = Join-Path $repoRoot '.github\isr-verifier-registry.json'
$tierRunnerPath = Join-Path $repoRoot '.github\scripts\isr-run-tiered-verification.ps1'

$violations = New-Object 'System.Collections.Generic.List[string]'

function Add-Violation {
    param([string]$Message)
    $violations.Add($Message) | Out-Null
}

function Assert-HasField {
    param(
        [Parameter(Mandatory = $true)]$Object,
        [Parameter(Mandatory = $true)][string]$Field,
        [Parameter(Mandatory = $true)][string]$Context
    )

    if ($null -eq $Object.PSObject.Properties[$Field]) {
        Add-Violation "$Context missing required field: $Field"
        return
    }

    $value = $Object.$Field
    if ($null -eq $value) {
        Add-Violation "$Context missing required field: $Field"
        return
    }

    if ($value -is [string] -and [string]::IsNullOrWhiteSpace($value)) {
        Add-Violation "$Context missing required field: $Field"
    }
}

function Assert-ValidExpiry {
    param(
        [Parameter(Mandatory = $true)][string]$Value,
        [Parameter(Mandatory = $true)][string]$Context
    )

    try {
        $expiry = [datetime]::ParseExact($Value, 'yyyy-MM-dd', [System.Globalization.CultureInfo]::InvariantCulture)
        if ((Get-Date).Date -gt $expiry.Date) {
            Add-Violation "$Context expired: expiry=$Value"
        }
    }
    catch {
        Add-Violation "$Context has invalid expiry format: '$Value' (expected yyyy-MM-dd)"
    }
}

if (-not (Test-Path -LiteralPath $contractRegistryPath)) {
    Add-Violation "Missing contract registry: $contractRegistryPath"
}

if (-not (Test-Path -LiteralPath $verifierRegistryPath)) {
    Add-Violation "Missing verifier registry: $verifierRegistryPath"
}

if (-not (Test-Path -LiteralPath $tierRunnerPath)) {
    Add-Violation "Missing tier runner script: $tierRunnerPath"
}

$contractRegistry = $null
$verifierRegistry = $null
$tierRunnerText = $null

if (Test-Path -LiteralPath $tierRunnerPath) {
    $tierRunnerText = Get-Content -LiteralPath $tierRunnerPath -Raw -Encoding UTF8
}

if (Test-Path -LiteralPath $contractRegistryPath) {
    $contractRegistry = Get-Content -LiteralPath $contractRegistryPath -Raw -Encoding UTF8 | ConvertFrom-Json
    if ("$($contractRegistry.schema)" -ne 'isr_contract_registry_v1') {
        Add-Violation "Contract registry schema mismatch: expected=isr_contract_registry_v1 actual=$($contractRegistry.schema)"
    }

    foreach ($field in @('version', 'owner', 'issue', 'rationale', 'expiry', 'source', 'contracts')) {
        Assert-HasField -Object $contractRegistry -Field $field -Context 'contractRegistry'
    }

    Assert-ValidExpiry -Value "$($contractRegistry.expiry)" -Context 'contractRegistry'

    $contracts = @($contractRegistry.contracts)
    if ($contracts.Count -eq 0) {
        Add-Violation 'contractRegistry.contracts must be non-empty'
    }

    $seenContractIds = @{}
    foreach ($entry in $contracts) {
        foreach ($field in @('id', 'category', 'description', 'owner', 'phase', 'status')) {
            Assert-HasField -Object $entry -Field $field -Context 'contractRegistry.contracts[]'
        }

        $id = "$($entry.id)"
        if (-not [string]::IsNullOrWhiteSpace($id)) {
            if ($seenContractIds.ContainsKey($id)) {
                Add-Violation "Duplicate contract id detected: $id"
            }
            else {
                $seenContractIds[$id] = $true
            }
        }
    }

    $requiredContractIds = @(
        'publication-authority-contract',
        'epoch-generation-mapping-contract',
        'partial-publication-zero-contract',
        'hash-authority-prohibition',
        'shadow-compare-cadence-contract',
        'retire-lifecycle-state-contract',
        'governance-consistency-contract'
    )

    foreach ($requiredId in $requiredContractIds) {
        if (-not $seenContractIds.ContainsKey($requiredId)) {
            Add-Violation "Required contract id is missing from contract registry: $requiredId"
        }
    }
}

if (Test-Path -LiteralPath $verifierRegistryPath) {
    $verifierRegistry = Get-Content -LiteralPath $verifierRegistryPath -Raw -Encoding UTF8 | ConvertFrom-Json
    if ("$($verifierRegistry.schema)" -ne 'isr_verifier_registry_v1') {
        Add-Violation "Verifier registry schema mismatch: expected=isr_verifier_registry_v1 actual=$($verifierRegistry.schema)"
    }

    foreach ($field in @('version', 'owner', 'issue', 'rationale', 'expiry', 'source', 'tiers', 'dependencies')) {
        Assert-HasField -Object $verifierRegistry -Field $field -Context 'verifierRegistry'
    }

    Assert-ValidExpiry -Value "$($verifierRegistry.expiry)" -Context 'verifierRegistry'

    $tierNames = @('pr', 'nightly', 'release')
    $allVerifiers = New-Object 'System.Collections.Generic.HashSet[string]'
    foreach ($tierName in $tierNames) {
        $tierList = @($verifierRegistry.tiers.$tierName)
        if ($tierList.Count -eq 0) {
            Add-Violation "verifierRegistry.tiers.$tierName must be non-empty"
            continue
        }

        foreach ($verifier in $tierList) {
            $name = "$verifier"
            if ([string]::IsNullOrWhiteSpace($name)) {
                Add-Violation "verifierRegistry.tiers.$tierName contains empty verifier name"
                continue
            }
            [void]$allVerifiers.Add($name)
        }
    }

    $requiredVerifierNames = @(
        'publication-single-source-verifier',
        'publication-monotonicity-verifier',
        'publication-visibility-monotonicity-verifier',
        'publication-epoch-generation-mapping-verifier',
        'shadow-compare-contract-verifier',
        'shadow-compare-cadence-verifier',
        'retire-lifecycle-state-verifier',
        'semantic-migration-compatibility-verifier',
        'governance-consistency-verifier'
    )

    foreach ($requiredVerifier in $requiredVerifierNames) {
        if (-not $allVerifiers.Contains($requiredVerifier)) {
            Add-Violation "Required verifier is missing from tier registry: $requiredVerifier"
        }
    }

    $dependencyMap = @{}
    foreach ($entry in @($verifierRegistry.dependencies)) {
        Assert-HasField -Object $entry -Field 'verifier' -Context 'verifierRegistry.dependencies[]'
        Assert-HasField -Object $entry -Field 'dependsOn' -Context 'verifierRegistry.dependencies[]'

        $verifierName = "$($entry.verifier)"
        $dependsOnList = @($entry.dependsOn)

        if ($dependsOnList.Count -eq 0) {
            Add-Violation "Dependency entry has empty dependsOn: verifier=$verifierName"
            continue
        }

        if ($dependencyMap.ContainsKey($verifierName)) {
            Add-Violation "Duplicate dependency entry for verifier: $verifierName"
            continue
        }

        $dependencyMap[$verifierName] = $dependsOnList

        foreach ($dep in $dependsOnList) {
            $depName = "$dep"
            if ($depName -eq $verifierName) {
                Add-Violation "Self dependency detected: verifier=$verifierName"
            }
            if (-not $allVerifiers.Contains($depName)) {
                Add-Violation "Dependency target not found in tier lists: verifier=$verifierName dependsOn=$depName"
            }
        }

        if (-not $allVerifiers.Contains($verifierName)) {
            Add-Violation "Dependency verifier not found in tier lists: verifier=$verifierName"
        }
    }

    # Dependency cycle check (DAG contract)
    $visiting = New-Object 'System.Collections.Generic.HashSet[string]'
    $visited = New-Object 'System.Collections.Generic.HashSet[string]'
    $hasCycle = $false

    function Test-DependencyCycle {
        param([string]$Node)

        if ($visited.Contains($Node)) {
            return
        }
        if ($visiting.Contains($Node)) {
            $script:hasCycle = $true
            return
        }

        [void]$visiting.Add($Node)
        if ($dependencyMap.ContainsKey($Node)) {
            foreach ($next in @($dependencyMap[$Node])) {
                Test-DependencyCycle -Node "$next"
            }
        }
        [void]$visiting.Remove($Node)
        [void]$visited.Add($Node)
    }

    foreach ($node in $dependencyMap.Keys) {
        Test-DependencyCycle -Node "$node"
    }

    if ($hasCycle) {
        Add-Violation 'Verifier dependency DAG contains a cycle'
    }

    if ($null -ne $tierRunnerText) {
        $wiringContracts = @(
            [ordered]@{ verifier = 'publication-single-source-verifier'; script = '.github/scripts/isr-verify-publication-single-path.ps1' },
            [ordered]@{ verifier = 'publication-visibility-monotonicity-verifier'; script = '.github/scripts/isr-verify-v8.ps1' },
            [ordered]@{ verifier = 'retire-lifecycle-state-verifier'; script = '.github/scripts/isr-verify-retire-lifecycle-state.ps1' },
            [ordered]@{ verifier = 'shadow-compare-contract-verifier'; script = '.github/scripts/isr-verify-shadow-compare-contract.ps1' },
            [ordered]@{ verifier = 'shadow-compare-cadence-verifier'; script = '.github/scripts/isr-verify-shadow-compare-cadence.ps1' },
            [ordered]@{ verifier = 'evidence-hierarchy-verifier'; script = '.github/scripts/isr-verify-evidence-hierarchy.ps1' },
            [ordered]@{ verifier = 'projection-austerity-verifier'; script = '.github/scripts/isr-verify-projection-austerity.ps1' },
            [ordered]@{ verifier = 'projection-freshness-verifier'; script = '.github/scripts/isr-verify-projection-freshness.ps1' },
            [ordered]@{ verifier = 'semantic-migration-compatibility-verifier'; script = '.github/scripts/isr-verify-semantic-migration-compatibility.ps1' },
            [ordered]@{ verifier = 'governance-consistency-verifier'; script = '.github/scripts/isr-verify-governance-consistency.ps1' }
        )

        foreach ($contract in $wiringContracts) {
            $verifierName = [string]$contract.verifier
            $scriptPath = [string]$contract.script

            if ($allVerifiers.Contains($verifierName) -and (-not $tierRunnerText.Contains("'$scriptPath'"))) {
                Add-Violation "Verifier wiring missing in tier runner: verifier=$verifierName script=$scriptPath"
            }
        }
    }
}

$report = [ordered]@{
    schema               = 'governance_registries_report_v1'
    generatedAt          = (Get-Date -Format 'o')
    contractRegistryPath = $contractRegistryPath
    verifierRegistryPath = $verifierRegistryPath
    violations           = @($violations)
    ready                = ($violations.Count -eq 0)
}

$report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $reportPath -Encoding UTF8
Write-Host "[INFO] governance registries report written: $reportPath"

if ($violations.Count -gt 0) {
    foreach ($v in $violations) {
        Write-Host "[ERROR] $v"
    }
    throw "Governance registry verification failed. violations=$($violations.Count)"
}

Write-Host '[PASS] governance registries verification passed'
