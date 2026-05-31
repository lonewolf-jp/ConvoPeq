$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$schemaPath = Join-Path $repoRoot 'src\audioengine\ISRRuntimeSemanticSchema.h'
$tierRunnerPath = Join-Path $repoRoot '.github\scripts\isr-run-tiered-verification.ps1'
$shadowContractPath = Join-Path $repoRoot '.github\scripts\isr-verify-shadow-compare-contract.ps1'
$equivalencePath = Join-Path $repoRoot '.github\scripts\isr-verify-semantic-equivalence.ps1'
$evidenceDir = Join-Path $repoRoot 'evidence'
$reportPath = Join-Path $evidenceDir 'semantic_hash_drift_report.json'

if (-not (Test-Path -LiteralPath $evidenceDir)) {
    New-Item -ItemType Directory -Path $evidenceDir -Force | Out-Null
}

$violations = New-Object 'System.Collections.Generic.List[string]'

foreach ($path in @($schemaPath, $tierRunnerPath, $shadowContractPath, $equivalencePath)) {
    if (-not (Test-Path -LiteralPath $path)) {
        $violations.Add("Missing semantic hash drift target: $path") | Out-Null
    }
}

$schemaText = if (Test-Path -LiteralPath $schemaPath) { Get-Content -LiteralPath $schemaPath -Raw -Encoding UTF8 } else { '' }
$tierRunnerText = if (Test-Path -LiteralPath $tierRunnerPath) { Get-Content -LiteralPath $tierRunnerPath -Raw -Encoding UTF8 } else { '' }

$requiredHashFields = @(
    'generationSemanticHash',
    'topologyHash',
    'executionHash',
    'routingHash',
    'payloadHash',
    'publicationSemanticHash',
    'overlapSemanticHash',
    'retireSemanticHash'
)

foreach ($field in $requiredHashFields) {
    if (-not $schemaText.Contains($field)) {
        $violations.Add("RuntimeSemanticHash missing required field: $field") | Out-Null
    }
}

if (-not [regex]::IsMatch($schemaText, 'classifySemanticEquivalence\(')) {
    $violations.Add('Semantic equivalence classifier is missing from schema header') | Out-Null
}

if (-not $tierRunnerText.Contains('.github/scripts/isr-verify-shadow-compare-contract.ps1')) {
    $violations.Add('Tier runner must execute shadow compare contract verifier for drift governance') | Out-Null
}

if (-not $tierRunnerText.Contains('.github/scripts/isr-verify-semantic-equivalence.ps1')) {
    $violations.Add('Tier runner must execute semantic equivalence verifier for drift governance') | Out-Null
}

$report = [ordered]@{
    schema             = 'semantic_hash_drift_report_v1'
    generatedAt        = (Get-Date -Format 'o')
    schemaPath         = $schemaPath
    tierRunnerPath     = $tierRunnerPath
    requiredHashFields = $requiredHashFields
    checks             = [ordered]@{
        hashFieldsPresent            = (@($requiredHashFields | Where-Object { $schemaText.Contains($_) }).Count -eq $requiredHashFields.Count)
        hasEquivalenceClassifier     = [regex]::IsMatch($schemaText, 'classifySemanticEquivalence\(')
        tierHasShadowCompareContract = $tierRunnerText.Contains('.github/scripts/isr-verify-shadow-compare-contract.ps1')
        tierHasSemanticEquivalence   = $tierRunnerText.Contains('.github/scripts/isr-verify-semantic-equivalence.ps1')
    }
    violations         = @($violations)
    ready              = ($violations.Count -eq 0)
}

$report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $reportPath -Encoding UTF8
Write-Host "[INFO] report: $reportPath"
if ($violations.Count -gt 0) {
    foreach ($v in $violations) { Write-Host "[ERROR] $v" }
    throw 'semantic hash drift verification failed'
}

Write-Host '[PASS] semantic hash drift verification passed'
