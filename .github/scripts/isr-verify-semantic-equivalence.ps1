# isr-verify-semantic-equivalence.ps1
# §3.19.6 Semantic Equivalence Contract (SemanticEquivalenceVerifier)
# Verifies that RuntimeSemanticHash uses all 8 required hash dimensions,
# ensuring that semantic equivalence comparison is multi-dimensional
# and not collapsed to a single hash value.

$ErrorActionPreference = 'Stop'
$repoRoot    = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$evidenceDir = Join-Path $repoRoot 'evidence'
$reportPath  = Join-Path $evidenceDir 'isr_semantic_equivalence_report.json'

if (-not (Test-Path -LiteralPath $evidenceDir)) {
    New-Item -ItemType Directory -Path $evidenceDir -Force | Out-Null
}

$violations = [System.Collections.Generic.List[string]]::new()
function Add-Violation { param([string]$Msg); $violations.Add($Msg) | Out-Null }

$schemaHeader = Join-Path $repoRoot 'src\audioengine\ISRRuntimeSemanticSchema.h'
if (-not (Test-Path -LiteralPath $schemaHeader)) {
    Add-Violation "ISRRuntimeSemanticSchema.h not found: $schemaHeader"
}
else {
    $sc = Get-Content -LiteralPath $schemaHeader -Raw
    if (-not $sc.Contains('SemanticEquivalenceVerifier')) {
        Add-Violation 'SemanticEquivalenceVerifier not registered in kRequiredVerifierTable'
    }
    # All 8 required hash dimensions must be present in RuntimeSemanticHash
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
    $missingFields = @()
    foreach ($f in $requiredHashFields) {
        if (-not $sc.Contains($f)) {
            $missingFields += $f
        }
    }
    if ($missingFields.Count -gt 0) {
        Add-Violation "RuntimeSemanticHash missing hash dimensions: $($missingFields -join ', ')"
    }
    else {
        Write-Host "[INFO] All 8 hash dimensions confirmed in RuntimeSemanticHash"
    }
}

$report = [ordered]@{
    schema      = 'isr_semantic_equivalence_evidence_v1'
    generatedAt = (Get-Date -Format 'o')
    violations  = @($violations)
    ready       = ($violations.Count -eq 0)
}

$report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $reportPath -Encoding UTF8
Write-Host "[INFO] SemanticEquivalenceVerifier evidence written: $reportPath"

if ($violations.Count -gt 0) {
    foreach ($v in $violations) { Write-Host "[FAIL] $v" }
    throw "SemanticEquivalenceVerifier contract violation. violations=$($violations.Count)"
}
Write-Host '[PASS] SemanticEquivalenceVerifier contract verification passed'
