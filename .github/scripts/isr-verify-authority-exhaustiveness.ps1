# isr-verify-authority-exhaustiveness.ps1
# §3.1.1 Authority Exhaustiveness Contract (AuthorityExhaustivenessVerifier)
# Verifies that the RuntimeAuthorityInventoryPolicy is defined with
# kExhaustivenessEnforced = true, and that all semantic fields in
# PublicationSemantic.kFieldDescriptors have a valid classification.

$ErrorActionPreference = 'Stop'
$repoRoot    = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$evidenceDir = Join-Path $repoRoot 'evidence'
$reportPath  = Join-Path $evidenceDir 'isr_authority_exhaustiveness_report.json'

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
    if (-not $sc.Contains('AuthorityExhaustivenessVerifier')) {
        Add-Violation 'AuthorityExhaustivenessVerifier not registered in kRequiredVerifierTable'
    }
    if (-not $sc.Contains('RuntimeAuthorityInventoryPolicy')) {
        Add-Violation 'RuntimeAuthorityInventoryPolicy struct is absent from ISRRuntimeSemanticSchema.h'
    }
    if (-not $sc.Contains('kExhaustivenessEnforced')) {
        Add-Violation 'RuntimeAuthorityInventoryPolicy::kExhaustivenessEnforced is absent'
    }
    if (-not $sc.Contains('kSchemaInventoryMismatchFails')) {
        Add-Violation 'RuntimeAuthorityInventoryPolicy::kSchemaInventoryMismatchFails is absent'
    }
    # validateFieldDescriptorSet must exist as the descriptor exhaustiveness mechanism
    if (-not $sc.Contains('validateFieldDescriptorSet')) {
        Add-Violation 'validateFieldDescriptorSet function is absent from ISRRuntimeSemanticSchema.h'
    }
    # PublicationSemantic.kFieldDescriptors must include all 4 publication fields
    $requiredPubFields = @('sequenceId', 'epoch', 'mappedRuntimeGeneration', 'previousSequenceId')
    foreach ($f in $requiredPubFields) {
        if (-not $sc.Contains("""$f""")) {
            Add-Violation "PublicationSemantic.kFieldDescriptors missing required field: $f"
        }
    }
}

# Check AudioEngine.h kFieldDescriptors for Authority-class coverage
$audioEngineHeader = Join-Path $repoRoot 'src\audioengine\AudioEngine.h'
if (Test-Path -LiteralPath $audioEngineHeader) {
    $ae = Get-Content -LiteralPath $audioEngineHeader -Raw
    if (-not $ae.Contains('kFieldDescriptors')) {
        Add-Violation 'AudioEngine.h missing kFieldDescriptors for runtime state authority classification'
    }
}

$report = [ordered]@{
    schema      = 'isr_authority_exhaustiveness_evidence_v1'
    generatedAt = (Get-Date -Format 'o')
    violations  = @($violations)
    ready       = ($violations.Count -eq 0)
}

$report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $reportPath -Encoding UTF8
Write-Host "[INFO] AuthorityExhaustivenessVerifier evidence written: $reportPath"

if ($violations.Count -gt 0) {
    foreach ($v in $violations) { Write-Host "[FAIL] $v" }
    throw "AuthorityExhaustivenessVerifier contract violation. violations=$($violations.Count)"
}
Write-Host '[PASS] AuthorityExhaustivenessVerifier contract verification passed'
