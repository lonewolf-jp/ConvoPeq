# isr-verify-runtime-world-identity.ps1
# §3.12.1 RuntimeWorld Identity Contract (RuntimeWorldIdentityVerifier)
# Verifies that worldId, generation, and semanticHash are assigned
# in the world construction/publication path.

$ErrorActionPreference = 'Stop'
$repoRoot    = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$evidenceDir = Join-Path $repoRoot 'evidence'
$reportPath  = Join-Path $evidenceDir 'isr_runtime_world_identity_report.json'

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
    if (-not $sc.Contains('RuntimeWorldIdentityVerifier')) {
        Add-Violation 'RuntimeWorldIdentityVerifier not registered in kRequiredVerifierTable'
    }
    # RuntimeSemanticHash must contain generationSemanticHash (world identity dimension)
    if (-not $sc.Contains('generationSemanticHash')) {
        Add-Violation 'RuntimeSemanticHash missing generationSemanticHash identity dimension'
    }
    # TopologySemantic must carry runtimeUuid as identity field
    if (-not $sc.Contains('runtimeUuid')) {
        Add-Violation 'TopologySemantic missing runtimeUuid identity field'
    }
}

# Contract: world identity fields must be assigned in buildRuntimePublishWorld
# (RuntimeBuilder.cpp), and commit path must consume world.generation /
# world.publication.sequenceId (AudioEngine.Commit.cpp).
$runtimeBuilder = Join-Path $repoRoot 'src\audioengine\RuntimeBuilder.cpp'
if (-not (Test-Path -LiteralPath $runtimeBuilder)) {
    Add-Violation "RuntimeBuilder.cpp not found: $runtimeBuilder"
}
else {
    $rb = Get-Content -LiteralPath $runtimeBuilder -Raw -Encoding UTF8
    if (-not $rb.Contains('buildRuntimePublishWorld(')) {
        Add-Violation 'RuntimeBuilder.cpp missing buildRuntimePublishWorld function'
    }
    if (-not $rb.Contains('worldOwner->worldId =')) {
        Add-Violation 'buildRuntimePublishWorld does not assign worldOwner->worldId'
    }
    if (-not $rb.Contains('worldOwner->generation =')) {
        Add-Violation 'buildRuntimePublishWorld does not assign worldOwner->generation'
    }
    if (-not $rb.Contains('worldOwner->publication.sequenceId =')) {
        Add-Violation 'buildRuntimePublishWorld does not assign worldOwner->publication.sequenceId'
    }
    if (-not $rb.Contains('worldOwner->semanticHash.generationSemanticHash =')) {
        Add-Violation 'buildRuntimePublishWorld does not compute semanticHash.generationSemanticHash'
    }
    if (-not $rb.Contains('worldOwner->freeze()')) {
        Add-Violation 'buildRuntimePublishWorld must freeze worldOwner before publish'
    }
}

$commitCpp = Join-Path $repoRoot 'src\audioengine\AudioEngine.Commit.cpp'
if (-not (Test-Path -LiteralPath $commitCpp)) {
    Add-Violation "AudioEngine.Commit.cpp not found: $commitCpp"
}
else {
    $cc = Get-Content -LiteralPath $commitCpp -Raw -Encoding UTF8
    if (-not $cc.Contains('onRuntimePublishedNonRt(const RuntimePublishWorld& world)')) {
        Add-Violation 'AudioEngine.Commit.cpp missing onRuntimePublishedNonRt(world) commit path'
    }
    if (-not $cc.Contains('world.generation')) {
        Add-Violation 'Commit path does not consume world.generation identity'
    }
    if (-not $cc.Contains('world.publication.sequenceId')) {
        Add-Violation 'Commit path does not consume world.publication.sequenceId identity'
    }
    if (-not $cc.Contains('lastCommittedRuntimeGeneration_')) {
        Add-Violation 'AudioEngine.Commit.cpp does not enforce generation monotonicity contract'
    }
}

$report = [ordered]@{
    schema      = 'isr_runtime_world_identity_evidence_v1'
    generatedAt = (Get-Date -Format 'o')
    violations  = @($violations)
    ready       = ($violations.Count -eq 0)
}

$report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $reportPath -Encoding UTF8
Write-Host "[INFO] RuntimeWorldIdentityVerifier evidence written: $reportPath"

if ($violations.Count -gt 0) {
    foreach ($v in $violations) { Write-Host "[FAIL] $v" }
    throw "RuntimeWorldIdentityVerifier contract violation. violations=$($violations.Count)"
}
Write-Host '[PASS] RuntimeWorldIdentityVerifier contract verification passed'
