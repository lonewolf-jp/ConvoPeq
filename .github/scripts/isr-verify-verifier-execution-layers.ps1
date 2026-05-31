$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$evidenceDir = Join-Path $repoRoot 'evidence'
$reportPath = Join-Path $evidenceDir 'verifier_execution_layers_report.json'

if (-not (Test-Path -LiteralPath $evidenceDir)) {
    New-Item -ItemType Directory -Path $evidenceDir -Force | Out-Null
}

$policyPath = Join-Path $repoRoot '.github\isr-verifier-execution-layers.json'
$schemaHeaderPath = Join-Path $repoRoot 'src\audioengine\ISRRuntimeSemanticSchema.h'
$tierRunnerPath = Join-Path $repoRoot '.github\scripts\isr-run-tiered-verification.ps1'
$unitSchemaTestPath = Join-Path $repoRoot 'src\tests\RuntimeSemanticSchemaValidationTests.cpp'
$unitRetireTestPath = Join-Path $repoRoot 'src\tests\RetireGraceSemanticsTests.cpp'

$violations = New-Object 'System.Collections.Generic.List[string]'

function Add-Violation {
    param([string]$Message)
    $violations.Add($Message) | Out-Null
}

foreach ($requiredPath in @($policyPath, $schemaHeaderPath, $tierRunnerPath, $unitSchemaTestPath, $unitRetireTestPath)) {
    if (-not (Test-Path -LiteralPath $requiredPath)) {
        Add-Violation "Missing required file: $requiredPath"
    }
}

if ($violations.Count -gt 0) {
    $report = [ordered]@{
        schema      = 'verifier_execution_layers_report_v1'
        generatedAt = (Get-Date -Format 'o')
        ready       = $false
        violations  = @($violations)
    }
    $report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $reportPath -Encoding UTF8
    throw "Verifier execution-layer wiring failed before validation. violations=$($violations.Count)"
}

$policy = Get-Content -LiteralPath $policyPath -Raw -Encoding UTF8 | ConvertFrom-Json
$schemaHeader = Get-Content -LiteralPath $schemaHeaderPath -Raw -Encoding UTF8
$tierRunner = Get-Content -LiteralPath $tierRunnerPath -Raw -Encoding UTF8
$unitSchemaTest = Get-Content -LiteralPath $unitSchemaTestPath -Raw -Encoding UTF8
$unitRetireTest = Get-Content -LiteralPath $unitRetireTestPath -Raw -Encoding UTF8

$unitContractTestContents = @{}
foreach ($unitTestPath in @($policy.unitContracts.tests)) {
    $resolvedPath = Join-Path $repoRoot "$unitTestPath"
    if (-not (Test-Path -LiteralPath $resolvedPath)) {
        Add-Violation "Unit contract test file not found: $unitTestPath"
        continue
    }

    $unitContractTestContents[$unitTestPath] = Get-Content -LiteralPath $resolvedPath -Raw -Encoding UTF8
}

foreach ($field in @('schema', 'version', 'owner', 'issue', 'rationale', 'expiry', 'compileTimeContracts', 'unitContracts', 'verifiers')) {
    if ($null -eq $policy.$field) {
        Add-Violation "Execution-layer policy missing required field: $field"
        continue
    }

    if ($field -eq 'verifiers') {
        if (@($policy.verifiers).Count -eq 0) {
            Add-Violation 'Execution-layer policy field verifiers must be non-empty'
        }
    }
    elseif ($field -in @('compileTimeContracts', 'unitContracts')) {
        # object presence already validated above
        continue
    }
    elseif ([string]::IsNullOrWhiteSpace("$($policy.$field)")) {
        Add-Violation "Execution-layer policy missing required field: $field"
    }
}

if ("$($policy.schema)" -ne 'isr_verifier_execution_layers_v1') {
    Add-Violation "Execution-layer policy schema mismatch: expected=isr_verifier_execution_layers_v1 actual=$($policy.schema)"
}

try {
    $expiry = [datetime]::ParseExact("$($policy.expiry)", 'yyyy-MM-dd', [System.Globalization.CultureInfo]::InvariantCulture)
    if ((Get-Date).Date -gt $expiry.Date) {
        Add-Violation "Execution-layer policy expired: expiry=$($policy.expiry)"
    }
}
catch {
    Add-Violation "Execution-layer policy has invalid expiry format: '$($policy.expiry)' (expected yyyy-MM-dd)"
}

$verifiers = @($policy.verifiers)
if ($verifiers.Count -ne 37) {
    Add-Violation "Execution-layer policy requires exactly 37 verifier entries. actual=$($verifiers.Count)"
}

$headerTableMatches = [regex]::Matches($schemaHeader, '\{"(?<name>[A-Za-z0-9]+)",\s*VerifierSeverity::(?<severity>[A-Za-z]+)\}')
$headerVerifierMap = @{}
foreach ($match in $headerTableMatches) {
    $name = $match.Groups['name'].Value
    $severity = $match.Groups['severity'].Value
    if ($headerVerifierMap.ContainsKey($name)) {
        Add-Violation "Header verifier table contains duplicate verifier name: $name"
    }
    else {
        $headerVerifierMap[$name] = $severity
    }
}

if ($headerVerifierMap.Count -ne 37) {
    Add-Violation "Header verifier table requires exactly 37 entries. actual=$($headerVerifierMap.Count)"
}

$policyVerifierMap = @{}
foreach ($entry in $verifiers) {
    foreach ($requiredField in @('name', 'severity', 'integrationScript', 'soakScript', 'gateTier')) {
        if ($null -eq $entry.$requiredField -or [string]::IsNullOrWhiteSpace("$($entry.$requiredField)")) {
            Add-Violation "Verifier policy entry missing required field: $requiredField"
        }
    }

    $name = "$($entry.name)"
    $severity = "$($entry.severity)"
    $gateTier = "$($entry.gateTier)"

    if ($policyVerifierMap.ContainsKey($name)) {
        Add-Violation "Execution-layer policy contains duplicate verifier entry: $name"
    }
    else {
        $policyVerifierMap[$name] = $entry
    }

    if ($severity -notin @('Warning', 'Error', 'Fatal')) {
        Add-Violation "Verifier policy has invalid severity: verifier=$name severity=$severity"
    }

    if ($gateTier -notin @('pr', 'nightly', 'release')) {
        Add-Violation "Verifier policy has invalid gateTier: verifier=$name gateTier=$gateTier"
    }

    $integrationScript = "$($entry.integrationScript)"
    $soakScript = "$($entry.soakScript)"
    if (-not (Test-Path -LiteralPath (Join-Path $repoRoot $integrationScript))) {
        Add-Violation "Integration script not found: verifier=$name path=$integrationScript"
    }
    if (-not (Test-Path -LiteralPath (Join-Path $repoRoot $soakScript))) {
        Add-Violation "Soak script not found: verifier=$name path=$soakScript"
    }
}

foreach ($headerVerifierName in $headerVerifierMap.Keys) {
    if (-not $policyVerifierMap.ContainsKey($headerVerifierName)) {
        Add-Violation "Execution-layer policy missing verifier from compile-time table: $headerVerifierName"
        continue
    }

    $headerSeverity = "$($headerVerifierMap[$headerVerifierName])"
    $policySeverity = "$($policyVerifierMap[$headerVerifierName].severity)"
    if ($headerSeverity -ne $policySeverity) {
        Add-Violation "Severity mismatch between compile-time table and execution-layer policy: verifier=$headerVerifierName header=$headerSeverity policy=$policySeverity"
    }
}

foreach ($policyVerifierName in $policyVerifierMap.Keys) {
    if (-not $headerVerifierMap.ContainsKey($policyVerifierName)) {
        Add-Violation "Execution-layer policy contains verifier not present in compile-time table: $policyVerifierName"
    }
}

foreach ($requiredAssert in @('kRequiredVerifierTable.size() == 37', 'validateVerifierTable()')) {
    if (-not $schemaHeader.Contains($requiredAssert)) {
        Add-Violation "Compile-time wiring missing static assert contract: $requiredAssert"
    }
}

if ($null -eq $policy.unitContracts.requiredUnitSymbols -or @($policy.unitContracts.requiredUnitSymbols).Count -eq 0) {
    Add-Violation 'Execution-layer policy requires non-empty unitContracts.requiredUnitSymbols'
}
else {
    foreach ($requiredUnitSymbol in @($policy.unitContracts.requiredUnitSymbols)) {
        $symbolText = "$requiredUnitSymbol"
        if ([string]::IsNullOrWhiteSpace($symbolText)) {
            Add-Violation 'Execution-layer policy contains empty requiredUnitSymbols entry'
            continue
        }

        $found = $false
        foreach ($content in $unitContractTestContents.Values) {
            if ($content.Contains($symbolText)) {
                $found = $true
                break
            }
        }

        if (-not $found) {
            Add-Violation "Unit wiring missing required symbol from unitContracts.requiredUnitSymbols: $symbolText"
        }
    }
}

foreach ($entry in $verifiers) {
    $name = "$($entry.name)"
    $integrationScript = "'$($entry.integrationScript)'"
    $soakScript = "'$($entry.soakScript)'"
    $gateTier = "$($entry.gateTier)"

    if (-not $tierRunner.Contains($integrationScript)) {
        Add-Violation "Integration-layer wiring missing in tier runner: verifier=$name script=$integrationScript"
    }

    if (-not $tierRunner.Contains($soakScript)) {
        Add-Violation "Soak-layer wiring missing in tier runner: verifier=$name script=$soakScript"
    }

    if ($gateTier -eq 'release' -and -not $tierRunner.Contains('exhaustiveAdditionalScripts')) {
        Add-Violation "Gate-tier wiring mismatch: verifier=$name gateTier=release requires exhaustive tier definition"
    }
}

$report = [ordered]@{
    schema            = 'verifier_execution_layers_report_v1'
    generatedAt       = (Get-Date -Format 'o')
    policyPath        = $policyPath
    compileTimeHeader = $schemaHeaderPath
    unitTests         = @($unitSchemaTestPath, $unitRetireTestPath)
    tierRunner        = $tierRunnerPath
    verifierCount     = $verifiers.Count
    violations        = @($violations)
    ready             = ($violations.Count -eq 0)
}

$report | ConvertTo-Json -Depth 16 | Set-Content -LiteralPath $reportPath -Encoding UTF8
Write-Host "[INFO] verifier execution layers report written: $reportPath"

if ($violations.Count -gt 0) {
    foreach ($v in $violations) {
        Write-Host "[ERROR] $v"
    }
    throw "Verifier execution-layer wiring verification failed. violations=$($violations.Count)"
}

Write-Host '[PASS] verifier execution-layer wiring verification passed'
