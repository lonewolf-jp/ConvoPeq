# isr-verify-runtime-admission.ps1
# §3.19.4 Runtime Admission Contract (RuntimeAdmissionVerifier)
# Verifies the 4-stage admission model is enforced in runPublicationPrecheckNonRt:
#   Stage 1: Completeness (schema version, descriptor set)
#   Stage 2: Validity (routing, execution semantic validators)
#   Stage 3: Admission gate (projection freshness, freeze/seal)
#   Stage 4: Publish (only if stages 1-3 pass)

$ErrorActionPreference = 'Stop'
$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$evidenceDir = Join-Path $repoRoot 'evidence'
$reportPath = Join-Path $evidenceDir 'isr_runtime_admission_report.json'

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
    if (-not $sc.Contains('RuntimeAdmissionVerifier')) {
        Add-Violation 'RuntimeAdmissionVerifier not registered in kRequiredVerifierTable'
    }
}

$commitCpp = Join-Path $repoRoot 'src\audioengine\AudioEngine.Commit.cpp'
if (-not (Test-Path -LiteralPath $commitCpp)) {
    Add-Violation "AudioEngine.Commit.cpp not found: $commitCpp"
}
else {
    $cc = Get-Content -LiteralPath $commitCpp -Raw
    # Stage 1 completeness
    if (-not $cc.Contains('runPublicationPrecheckNonRt')) {
        Add-Violation 'runPublicationPrecheckNonRt is absent from AudioEngine.Commit.cpp'
    }
    if (-not $cc.Contains('validateDescriptorSet')) {
        Add-Violation 'Stage-1 completeness: validateDescriptorSet call absent from precheck'
    }
    if (-not $cc.Contains('kRuntimeSemanticSchemaVersion')) {
        Add-Violation 'Stage-1 completeness: schema version check absent from precheck'
    }
    # Stage 2 validity
    if (-not $cc.Contains('isValidRoutingSemantic')) {
        Add-Violation 'Stage-2 validity: isValidRoutingSemantic call absent from precheck'
    }
    if (-not $cc.Contains('isValidExecutionSemantic')) {
        Add-Violation 'Stage-2 validity: isValidExecutionSemantic call absent from precheck'
    }
    # Stage 3 admission gate
    if (-not $cc.Contains('projectionFreshness')) {
        Add-Violation 'Stage-3 admission: projectionFreshness check absent from precheck'
    }
    if (-not $cc.Contains('isFrozen')) {
        Add-Violation 'Stage-3 admission: isFrozen check absent from precheck'
    }

    if (-not $cc.Contains('validateSemanticCompleteness(world)')) {
        Add-Violation 'Stage-1 completeness: validateSemanticCompleteness(world) call absent from precheck'
    }
    if (-not $cc.Contains('acceptsRuntimePublication()')) {
        Add-Violation 'Stage-3 admission: acceptsRuntimePublication() gate absent from precheck'
    }
    if (-not $cc.Contains('isSealedRecursively')) {
        Add-Violation 'Stage-3 admission: isSealedRecursively check absent from precheck'
    }
    if (-not $cc.Contains('transitionSemanticTransactionState(semanticTransactionState_, convo::isr::SemanticTransactionState::Validated)')) {
        Add-Violation 'Transaction stage transition to Validated is absent'
    }
    if (-not $cc.Contains('transitionSemanticTransactionState(semanticTransactionState_, convo::isr::SemanticTransactionState::Committed)')) {
        Add-Violation 'Transaction stage transition to Committed is absent'
    }
    if (-not $cc.Contains('SemanticTransactionState::Rejected')) {
        Add-Violation 'Reject path transition to SemanticTransactionState::Rejected is absent'
    }

    $precheckSignature = 'bool AudioEngine::runPublicationPrecheckNonRt(const RuntimePublishWorld& world) noexcept'
    $sigIndex = $cc.IndexOf($precheckSignature, [System.StringComparison]::Ordinal)
    if ($sigIndex -lt 0) {
        Add-Violation 'runPublicationPrecheckNonRt signature not found'
    }
    else {
        $nextFunctionMarker = 'void AudioEngine::onRuntimePublishedNonRt(const RuntimePublishWorld& world) noexcept'
        $endIndex = $cc.IndexOf($nextFunctionMarker, [System.StringComparison]::Ordinal)
        if ($endIndex -lt 0 -or $endIndex -le $sigIndex) {
            Add-Violation 'Unable to isolate runPublicationPrecheckNonRt function body'
        }
        else {
            $precheckBody = $cc.Substring($sigIndex, $endIndex - $sigIndex)

            $idxStage1 = $precheckBody.IndexOf('validateSemanticCompleteness(world)', [System.StringComparison]::Ordinal)
            $idxStage2 = $precheckBody.IndexOf('isValidRoutingSemantic(world.routing)', [System.StringComparison]::Ordinal)
            $idxStage3 = $precheckBody.IndexOf('acceptsRuntimePublication()', [System.StringComparison]::Ordinal)
            $idxStage4 = $precheckBody.IndexOf('SemanticTransactionState::Committed', [System.StringComparison]::Ordinal)

            if ($idxStage1 -lt 0 -or $idxStage2 -lt 0 -or $idxStage3 -lt 0 -or $idxStage4 -lt 0) {
                Add-Violation 'Could not detect all four admission stages in precheck body'
            }
            elseif (-not ($idxStage1 -lt $idxStage2 -and $idxStage2 -lt $idxStage3 -and $idxStage3 -lt $idxStage4)) {
                Add-Violation 'Admission stage order violation in runPublicationPrecheckNonRt (Stage1->Stage2->Stage3->Stage4)'
            }
        }
    }
}

$report = [ordered]@{
    schema      = 'isr_runtime_admission_evidence_v1'
    generatedAt = (Get-Date -Format 'o')
    violations  = @($violations)
    ready       = ($violations.Count -eq 0)
}

$report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $reportPath -Encoding UTF8
Write-Host "[INFO] RuntimeAdmissionVerifier evidence written: $reportPath"

if ($violations.Count -gt 0) {
    foreach ($v in $violations) { Write-Host "[FAIL] $v" }
    throw "RuntimeAdmissionVerifier contract violation. violations=$($violations.Count)"
}
Write-Host '[PASS] RuntimeAdmissionVerifier contract verification passed'
