# isr-verify-replacement-atomicity.ps1
# §3.13.1 Replacement Atomicity Contract (ReplacementAtomicityVerifier)
# Verifies that in the world replacement commit path, the new world is made
# visible (setActiveRuntimeDSP or equivalent) BEFORE the old world retire
# is initiated. This enforces atomic-visibility ordering.

$ErrorActionPreference = 'Stop'
$repoRoot    = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$evidenceDir = Join-Path $repoRoot 'evidence'
$reportPath  = Join-Path $evidenceDir 'isr_replacement_atomicity_report.json'

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
    if (-not $sc.Contains('ReplacementAtomicityVerifier')) {
        Add-Violation 'ReplacementAtomicityVerifier not registered in kRequiredVerifierTable'
    }
}

$commitCpp = Join-Path $repoRoot 'src\audioengine\AudioEngine.Commit.cpp'
if (-not (Test-Path -LiteralPath $commitCpp)) {
    Add-Violation "AudioEngine.Commit.cpp not found: $commitCpp"
}
else {
    $cc = Get-Content -LiteralPath $commitCpp -Raw
    # Both the activate and retire operations must be present in the commit path.
    $hasActivate = $cc.Contains('setActiveRuntimeDSP') -or $cc.Contains('publishWorld') -or $cc.Contains('executeCommit')
    $hasRetire   = $cc.Contains('retireRuntimeImmediately') -or $cc.Contains('onRuntimeRetiredNonRt')
    if (-not $hasActivate) {
        Add-Violation 'AudioEngine.Commit.cpp missing world activation call (setActiveRuntimeDSP/executeCommit)'
    }
    if (-not $hasRetire) {
        Add-Violation 'AudioEngine.Commit.cpp missing world retire call (retireRuntimeImmediately/onRuntimeRetiredNonRt)'
    }
    # Ordering check: activation token must appear before retire token in source
    if ($hasActivate -and $hasRetire) {
        $activateTokens = @('setActiveRuntimeDSP', 'executeCommit', 'onRuntimePublishedNonRt')
        $retireTokens   = @('retireRuntimeImmediately', 'onRuntimeRetiredNonRt')
        $activateIdx = [int]::MaxValue
        foreach ($t in $activateTokens) {
            $idx = $cc.IndexOf($t)
            if ($idx -ge 0 -and $idx -lt $activateIdx) { $activateIdx = $idx }
        }
        $retireIdx = [int]::MaxValue
        foreach ($t in $retireTokens) {
            $idx = $cc.IndexOf($t)
            if ($idx -ge 0 -and $idx -lt $retireIdx) { $retireIdx = $idx }
        }
        if ($activateIdx -eq [int]::MaxValue -or $retireIdx -eq [int]::MaxValue) {
            Add-Violation 'Could not locate activation or retire tokens for ordering verification'
        }
        elseif ($activateIdx -gt $retireIdx) {
            Add-Violation 'Replacement atomicity violated: retire token precedes activate token in commit path'
        }
    }
}

$report = [ordered]@{
    schema      = 'isr_replacement_atomicity_evidence_v1'
    generatedAt = (Get-Date -Format 'o')
    violations  = @($violations)
    ready       = ($violations.Count -eq 0)
}

$report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $reportPath -Encoding UTF8
Write-Host "[INFO] ReplacementAtomicityVerifier evidence written: $reportPath"

if ($violations.Count -gt 0) {
    foreach ($v in $violations) { Write-Host "[FAIL] $v" }
    throw "ReplacementAtomicityVerifier contract violation. violations=$($violations.Count)"
}
Write-Host '[PASS] ReplacementAtomicityVerifier contract verification passed'
