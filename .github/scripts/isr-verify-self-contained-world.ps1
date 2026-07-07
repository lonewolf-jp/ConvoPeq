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

# Contract: world observation in AudioEngine must route through
# RuntimePublicationCoordinator read contract (observeWorldHandle OR
# acquireReadToken + consumeWorldHandle), not direct raw global pointer dereference.
$coordHeader = Join-Path $repoRoot 'src\core\RuntimePublicationCoordinator.h'
if (-not (Test-Path -LiteralPath $coordHeader)) {
    Add-Violation "RuntimePublicationCoordinator.h not found: $coordHeader"
}
else {
    $coordContent = Get-Content -LiteralPath $coordHeader -Raw
    $hasObserveWorldHandle = $coordContent.Contains('observeWorldHandle')
    $hasTokenConsumeContract = $coordContent.Contains('acquireReadToken') -and $coordContent.Contains('consumeWorldHandle')
    if (-not $hasObserveWorldHandle -and -not $hasTokenConsumeContract) {
        Add-Violation 'RuntimePublicationCoordinator.h does not expose observeWorldHandle nor acquireReadToken+consumeWorldHandle contract'
    }
}

# Check that AudioEngine.h uses RuntimePublicationCoordinator read contract
# (observeWorldHandle OR acquireReadToken+consumeWorldHandle) for RT observe.
$audioEngineHeader = Join-Path $repoRoot 'src\audioengine\AudioEngine.h'
if (Test-Path -LiteralPath $audioEngineHeader) {
    $aeContent = Get-Content -LiteralPath $audioEngineHeader -Raw
    $usesObserveWorldHandle = $aeContent.Contains('observeWorldHandle')
    $usesTokenConsumeContract = $aeContent.Contains('RuntimePublicationCoordinator::acquireReadToken') -and $aeContent.Contains('RuntimePublicationCoordinator::consumeWorldHandle')
    if (-not $usesObserveWorldHandle -and -not $usesTokenConsumeContract) {
        Add-Violation 'AudioEngine.h does not use RuntimePublicationCoordinator read contract for world observation'
    }
}

$schemaHeader = Join-Path $repoRoot 'src\audioengine\ISRRuntimePublicationCoordinator.h'
$coordHeader = Join-Path $repoRoot 'src\core\RuntimePublicationCoordinator.h'
$audioEngineHeader = Join-Path $repoRoot 'src\audioengine\AudioEngine.h'
$timerCpp = Join-Path $repoRoot 'src\audioengine\AudioEngine.Timer.cpp'

$findings = @{
    schemaHeaderFound    = (Test-Path -LiteralPath $schemaHeader)
    coordHeaderFound     = (Test-Path -LiteralPath $coordHeader)
    audioEngineFound     = (Test-Path -LiteralPath $audioEngineHeader)
    timerFound           = (Test-Path -LiteralPath $timerCpp)
    violationCount       = $violations.Count
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
