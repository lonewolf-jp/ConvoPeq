# isr-verify-semantic-validity.ps1
# §3.19.3 Semantic Validity Contract (SemanticValidityVerifier)
# Verifies that isValidRoutingSemantic and isValidExecutionSemantic are both
# invoked within the publication precheck path before the admission gate.

$ErrorActionPreference = 'Stop'
$repoRoot    = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$evidenceDir = Join-Path $repoRoot 'evidence'
$reportPath  = Join-Path $evidenceDir 'isr_semantic_validity_report.json'

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
    if (-not $sc.Contains('SemanticValidityVerifier')) {
        Add-Violation 'SemanticValidityVerifier not registered in kRequiredVerifierTable'
    }
    if (-not $sc.Contains('isValidRoutingSemantic')) {
        Add-Violation 'isValidRoutingSemantic is not defined in ISRRuntimeSemanticSchema.h'
    }
    if (-not $sc.Contains('isValidExecutionSemantic')) {
        Add-Violation 'isValidExecutionSemantic is not defined in ISRRuntimeSemanticSchema.h'
    }
}

# Contract: both validators must be called in the commit/precheck path.
$commitCpp = Join-Path $repoRoot 'src\audioengine\AudioEngine.Commit.cpp'
if (-not (Test-Path -LiteralPath $commitCpp)) {
    Add-Violation "AudioEngine.Commit.cpp not found: $commitCpp"
}
else {
    $cc = Get-Content -LiteralPath $commitCpp -Raw -Encoding UTF8
    $precheckMatch = [regex]::Match(
        $cc,
        '(?s)\[\[nodiscard\]\]\s+bool\s+AudioEngine::runPublicationPrecheckNonRt\(.*?\)\s+noexcept\s*\{(?<body>.*?)\n\}\s*\nvoid\s+AudioEngine::onRuntimePublishedNonRt',
        [System.Text.RegularExpressions.RegexOptions]::Singleline
    )

    if (-not $precheckMatch.Success) {
        Add-Violation 'runPublicationPrecheckNonRt body extraction failed for semantic validity verification'
    }
    else {
        $precheckBody = $precheckMatch.Groups['body'].Value
        $routingIndex = $precheckBody.IndexOf('isValidRoutingSemantic')
        $executionIndex = $precheckBody.IndexOf('isValidExecutionSemantic')
        $admissionIndex = $precheckBody.IndexOf('acceptsRuntimePublication()')

        if ($routingIndex -lt 0) {
            Add-Violation 'AudioEngine.Commit.cpp does not call isValidRoutingSemantic in precheck'
        }
        if ($executionIndex -lt 0) {
            Add-Violation 'AudioEngine.Commit.cpp does not call isValidExecutionSemantic in precheck'
        }

        if ($admissionIndex -ge 0) {
            if ($routingIndex -ge 0 -and $routingIndex -gt $admissionIndex) {
                Add-Violation 'isValidRoutingSemantic must be evaluated before acceptsRuntimePublication()'
            }
            if ($executionIndex -ge 0 -and $executionIndex -gt $admissionIndex) {
                Add-Violation 'isValidExecutionSemantic must be evaluated before acceptsRuntimePublication()'
            }
        }
    }
}

$report = [ordered]@{
    schema      = 'isr_semantic_validity_evidence_v1'
    generatedAt = (Get-Date -Format 'o')
    violations  = @($violations)
    ready       = ($violations.Count -eq 0)
}

$report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $reportPath -Encoding UTF8
Write-Host "[INFO] SemanticValidityVerifier evidence written: $reportPath"

if ($violations.Count -gt 0) {
    foreach ($v in $violations) { Write-Host "[FAIL] $v" }
    throw "SemanticValidityVerifier contract violation. violations=$($violations.Count)"
}
Write-Host '[PASS] SemanticValidityVerifier contract verification passed'
