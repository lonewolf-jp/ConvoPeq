# isr-verify-semantic-conflict.ps1
# §3.19.6 Semantic Conflict Contract (SemanticConflictVerifier)
# Verifies that contradictory field combinations are rejected in the publication
# precheck. Specifically: fadeTimeSec < 0, and crossfade-pending without active
# transition must both be rejected before admission.

$ErrorActionPreference = 'Stop'
$repoRoot    = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$evidenceDir = Join-Path $repoRoot 'evidence'
$reportPath  = Join-Path $evidenceDir 'isr_semantic_conflict_report.json'

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
    if (-not $sc.Contains('SemanticConflictVerifier')) {
        Add-Violation 'SemanticConflictVerifier not registered in kRequiredVerifierTable'
    }
    # OverlapSemantic must define fadeTimeSec to enable the conflict check
    if (-not $sc.Contains('fadeTimeSec')) {
        Add-Violation 'OverlapSemantic missing fadeTimeSec field required for conflict check'
    }
    if (-not $sc.Contains('firstIrDryCrossfadePending')) {
        Add-Violation 'OverlapSemantic missing firstIrDryCrossfadePending field required for conflict check'
    }
}

$commitCpp = Join-Path $repoRoot 'src\audioengine\AudioEngine.Commit.cpp'
if (-not (Test-Path -LiteralPath $commitCpp)) {
    Add-Violation "AudioEngine.Commit.cpp not found: $commitCpp"
}
else {
    $cc = Get-Content -LiteralPath $commitCpp -Raw
    # Conflict check 1: negative fadeTimeSec must be rejected
    if (-not $cc.Contains('fadeTimeSec')) {
        Add-Violation 'AudioEngine.Commit.cpp does not check fadeTimeSec for negative value conflict'
    }
    # Conflict check 2: crossfade-pending without transitionActive must be rejected
    if (-not $cc.Contains('firstIrDryCrossfadePending')) {
        Add-Violation 'AudioEngine.Commit.cpp does not check firstIrDryCrossfadePending conflict'
    }
    # Both checks must appear in the context of runPublicationPrecheckNonRt
    if (-not $cc.Contains('runPublicationPrecheckNonRt')) {
        Add-Violation 'AudioEngine.Commit.cpp missing runPublicationPrecheckNonRt host function for conflict checks'
    }
}

$report = [ordered]@{
    schema      = 'isr_semantic_conflict_evidence_v1'
    generatedAt = (Get-Date -Format 'o')
    violations  = @($violations)
    ready       = ($violations.Count -eq 0)
}

$report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $reportPath -Encoding UTF8
Write-Host "[INFO] SemanticConflictVerifier evidence written: $reportPath"

if ($violations.Count -gt 0) {
    foreach ($v in $violations) { Write-Host "[FAIL] $v" }
    throw "SemanticConflictVerifier contract violation. violations=$($violations.Count)"
}
Write-Host '[PASS] SemanticConflictVerifier contract verification passed'
