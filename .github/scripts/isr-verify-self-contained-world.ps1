# isr-verify-self-contained-world.ps1
# §3.11.1 RuntimeWorld Self-Contained Contract (SelfContainedWorldVerifier)
# Verifies that the published RuntimeWorld observation path uses only
# RuntimePublicationCoordinator::observeWorldHandle and does not directly
# access raw mutable global/singleton state.

$ErrorActionPreference = 'Stop'
$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$evidenceDir = Join-Path $repoRoot 'evidence'
$reportPath = Join-Path $evidenceDir 'isr_self_contained_world_report.json'

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
    $schemaContent = Get-Content -LiteralPath $schemaHeader -Raw
    if (-not $schemaContent.Contains('SelfContainedWorldVerifier')) {
        Add-Violation 'SelfContainedWorldVerifier is not registered in kRequiredVerifierTable'
    }
}

# Contract: world observation in AudioEngine must route through observeWorldHandle,
# not direct raw global pointer dereference.
$coordHeader = Join-Path $repoRoot 'src\core\RuntimePublicationCoordinator.h'
if (-not (Test-Path -LiteralPath $coordHeader)) {
    Add-Violation "RuntimePublicationCoordinator.h not found: $coordHeader"
}
else {
    $coordContent = Get-Content -LiteralPath $coordHeader -Raw
    if (-not $coordContent.Contains('observeWorldHandle')) {
        Add-Violation 'observeWorldHandle is not defined in RuntimePublicationCoordinator.h'
    }
}

# Check that AudioEngine.h uses observeWorldHandle (not raw global dereference) for RT observe.
$audioEngineHeader = Join-Path $repoRoot 'src\audioengine\AudioEngine.h'
if (Test-Path -LiteralPath $audioEngineHeader) {
    $aeContent = Get-Content -LiteralPath $audioEngineHeader -Raw
    if (-not $aeContent.Contains('observeWorldHandle')) {
        Add-Violation 'AudioEngine.h does not use observeWorldHandle for world observation'
    }
}

$audioBlockCpp = Join-Path $repoRoot 'src\audioengine\AudioEngine.Processing.AudioBlock.cpp'
if (-not (Test-Path -LiteralPath $audioBlockCpp)) {
    Add-Violation "AudioEngine.Processing.AudioBlock.cpp not found: $audioBlockCpp"
}
else {
    $ab = Get-Content -LiteralPath $audioBlockCpp -Raw
    if ($ab.Contains('processWithSnapshot(')) {
        Add-Violation 'AudioBlock processing must not call processWithSnapshot (observe path collapse violation)'
    }
    if ($ab.Contains('captureAudioThreadParameterSnapshot(nullptr)')) {
        Add-Violation 'AudioBlock processing must not call captureAudioThreadParameterSnapshot(nullptr) (world authority fallback prohibited)'
    }
    if ($ab.Contains('consumeCrossfadePreparedSnapshot()')) {
        Add-Violation 'AudioBlock processing must not use consumeCrossfadePreparedSnapshot() as authority input'
    }
}

$blockDoubleCpp = Join-Path $repoRoot 'src\audioengine\AudioEngine.Processing.BlockDouble.cpp'
if (-not (Test-Path -LiteralPath $blockDoubleCpp)) {
    Add-Violation "AudioEngine.Processing.BlockDouble.cpp not found: $blockDoubleCpp"
}
else {
    $bd = Get-Content -LiteralPath $blockDoubleCpp -Raw
    if ($bd.Contains('processWithSnapshot(')) {
        Add-Violation 'BlockDouble processing must not call processWithSnapshot (observe path collapse violation)'
    }
    if ($bd.Contains('updateAudioThreadSnapshotFade(')) {
        Add-Violation 'BlockDouble processing must not call updateAudioThreadSnapshotFade after observe collapse'
    }
    if ($bd.Contains('captureAudioThreadParameterSnapshot(snapshotFrom)') -or $bd.Contains('captureAudioThreadParameterSnapshot(snapshotTo)')) {
        Add-Violation 'BlockDouble processing must not read snapshotFrom/snapshotTo parameter snapshots'
    }
    if ($bd.Contains('captureAudioThreadParameterSnapshot(nullptr)')) {
        Add-Violation 'BlockDouble processing must not call captureAudioThreadParameterSnapshot(nullptr) (world authority fallback prohibited)'
    }
    if ($bd.Contains('consumeCrossfadePreparedSnapshot()')) {
        Add-Violation 'BlockDouble processing must not use consumeCrossfadePreparedSnapshot() as authority input'
    }
}

$findings = @{
    schemaHeaderFound = (Test-Path -LiteralPath $schemaHeader)
    coordHeaderFound  = (Test-Path -LiteralPath $coordHeader)
    audioEngineFound  = (Test-Path -LiteralPath $audioEngineHeader)
    audioBlockFound   = (Test-Path -LiteralPath $audioBlockCpp)
    blockDoubleFound  = (Test-Path -LiteralPath $blockDoubleCpp)
    violationCount    = $violations.Count
}

$report = [ordered]@{
    schema      = 'isr_self_contained_world_evidence_v1'
    generatedAt = (Get-Date -Format 'o')
    violations  = @($violations)
    findings    = $findings
    ready       = ($violations.Count -eq 0)
}

$report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $reportPath -Encoding UTF8
Write-Host "[INFO] SelfContainedWorldVerifier evidence written: $reportPath"

if ($violations.Count -gt 0) {
    foreach ($v in $violations) { Write-Host "[FAIL] $v" }
    throw "SelfContainedWorldVerifier contract violation. violations=$($violations.Count)"
}
Write-Host '[PASS] SelfContainedWorldVerifier contract verification passed'
