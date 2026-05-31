# isr-verify-partial-semantic-update-prohibition.ps1
# §3.12.2 Partial Semantic Update Prohibition (PartialSemanticUpdateProhibitionVerifier)
# Verifies that the publication precheck enforces isFrozen() and isSealedRecursively()
# before allowing a world to be published. Any post-freeze field mutation is prohibited.

$ErrorActionPreference = 'Stop'
$repoRoot    = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$evidenceDir = Join-Path $repoRoot 'evidence'
$reportPath  = Join-Path $evidenceDir 'isr_partial_semantic_update_prohibition_report.json'

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
    if (-not $sc.Contains('PartialSemanticUpdateProhibitionVerifier')) {
        Add-Violation 'PartialSemanticUpdateProhibitionVerifier not registered in kRequiredVerifierTable'
    }
    # ImmutableAfterPublish mutability class must be defined (field-level prohibition)
    if (-not $sc.Contains('ImmutableAfterPublish')) {
        Add-Violation 'MutabilityClass::ImmutableAfterPublish is absent from ISRRuntimeSemanticSchema.h'
    }
}

# Contract: runPublicationPrecheckNonRt must call isFrozen() and isSealedRecursively().
$commitCpp = Join-Path $repoRoot 'src\audioengine\AudioEngine.Commit.cpp'
if (-not (Test-Path -LiteralPath $commitCpp)) {
    Add-Violation "AudioEngine.Commit.cpp not found: $commitCpp"
}
else {
    $cc = Get-Content -LiteralPath $commitCpp -Raw -Encoding UTF8
    if (-not $cc.Contains('runPublicationPrecheckNonRt')) {
        Add-Violation 'AudioEngine.Commit.cpp missing runPublicationPrecheckNonRt function'
    }

    $precheckMatch = [regex]::Match(
        $cc,
        '(?s)\[\[nodiscard\]\]\s+bool\s+AudioEngine::runPublicationPrecheckNonRt\(.*?\)\s+noexcept\s*\{(?<body>.*?)\n\}\s*\nvoid\s+AudioEngine::onRuntimePublishedNonRt',
        [System.Text.RegularExpressions.RegexOptions]::Singleline
    )

    if (-not $precheckMatch.Success) {
        Add-Violation 'runPublicationPrecheckNonRt body extraction failed'
    }
    else {
        $precheckBody = $precheckMatch.Groups['body'].Value
        if (-not $precheckBody.Contains('world.isFrozen()')) {
            Add-Violation 'runPublicationPrecheckNonRt does not call world.isFrozen()'
        }
        if (-not $precheckBody.Contains('world.isSealedRecursively()')) {
            Add-Violation 'runPublicationPrecheckNonRt does not call world.isSealedRecursively()'
        }

        $frozenIndex = $precheckBody.IndexOf('world.isFrozen()')
        $sealedIndex = $precheckBody.IndexOf('world.isSealedRecursively()')
        if ($frozenIndex -ge 0 -and $sealedIndex -ge 0 -and $sealedIndex -lt $frozenIndex) {
            Add-Violation 'runPublicationPrecheckNonRt must evaluate world.isFrozen() before world.isSealedRecursively()'
        }

        if ($precheckBody -match '(?m)\bworld\.[A-Za-z_][A-Za-z0-9_]*\s*=(?!=)') {
            Add-Violation 'runPublicationPrecheckNonRt must not mutate world.* fields (partial semantic update prohibited)'
        }
    }
}

$report = [ordered]@{
    schema      = 'isr_partial_semantic_update_prohibition_evidence_v1'
    generatedAt = (Get-Date -Format 'o')
    violations  = @($violations)
    ready       = ($violations.Count -eq 0)
}

$report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $reportPath -Encoding UTF8
Write-Host "[INFO] PartialSemanticUpdateProhibitionVerifier evidence written: $reportPath"

if ($violations.Count -gt 0) {
    foreach ($v in $violations) { Write-Host "[FAIL] $v" }
    throw "PartialSemanticUpdateProhibitionVerifier contract violation. violations=$($violations.Count)"
}
Write-Host '[PASS] PartialSemanticUpdateProhibitionVerifier contract verification passed'
