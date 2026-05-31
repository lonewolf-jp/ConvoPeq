# isr-verify-executor-snapshot-freshness.ps1
# §3.17.2 Executor Snapshot Freshness Contract (ExecutorSnapshotFreshnessVerifier)
# Verifies that the publication precheck enforces
# projectionFreshness.projectionGeneration == publication.mappedRuntimeGeneration.
# Drift (stale executor snapshot) must be rejected before admission.

$ErrorActionPreference = 'Stop'
$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$evidenceDir = Join-Path $repoRoot 'evidence'
$reportPath = Join-Path $evidenceDir 'isr_executor_snapshot_freshness_report.json'

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
    if (-not $sc.Contains('ExecutorSnapshotFreshnessVerifier')) {
        Add-Violation 'ExecutorSnapshotFreshnessVerifier not registered in kRequiredVerifierTable'
    }
    if (-not $sc.Contains('ExecutorSnapshotFreshnessPolicy')) {
        Add-Violation 'ExecutorSnapshotFreshnessPolicy struct is absent from ISRRuntimeSemanticSchema.h'
    }
    if (-not $sc.Contains('kGenerationMustMatch')) {
        Add-Violation 'ExecutorSnapshotFreshnessPolicy::kGenerationMustMatch is absent'
    }
    if (-not $sc.Contains('projectionGeneration')) {
        Add-Violation 'ProjectionFreshness missing projectionGeneration field'
    }
}

$commitCpp = Join-Path $repoRoot 'src\audioengine\AudioEngine.Commit.cpp'
if (-not (Test-Path -LiteralPath $commitCpp)) {
    Add-Violation "AudioEngine.Commit.cpp not found: $commitCpp"
}
else {
    $cc = Get-Content -LiteralPath $commitCpp -Raw
    # Precheck must compare projectionGeneration against mappedRuntimeGeneration
    if (-not $cc.Contains('projectionGeneration')) {
        Add-Violation 'AudioEngine.Commit.cpp does not check projectionGeneration in precheck'
    }
    if (-not $cc.Contains('mappedRuntimeGeneration')) {
        Add-Violation 'AudioEngine.Commit.cpp does not reference mappedRuntimeGeneration in precheck'
    }
    if (-not $cc.Contains('world.projectionFreshness.projectionGeneration != world.publication.mappedRuntimeGeneration')) {
        Add-Violation 'Precheck does not explicitly compare projectionGeneration against mappedRuntimeGeneration'
    }
}

$report = [ordered]@{
    schema      = 'isr_executor_snapshot_freshness_evidence_v1'
    generatedAt = (Get-Date -Format 'o')
    violations  = @($violations)
    ready       = ($violations.Count -eq 0)
}

$report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $reportPath -Encoding UTF8
Write-Host "[INFO] ExecutorSnapshotFreshnessVerifier evidence written: $reportPath"

if ($violations.Count -gt 0) {
    foreach ($v in $violations) { Write-Host "[FAIL] $v" }
    throw "ExecutorSnapshotFreshnessVerifier contract violation. violations=$($violations.Count)"
}
Write-Host '[PASS] ExecutorSnapshotFreshnessVerifier contract verification passed'
