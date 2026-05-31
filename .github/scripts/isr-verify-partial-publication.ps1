$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$commitSourcePath = Join-Path $repoRoot 'src\audioengine\AudioEngine.Commit.cpp'
$schemaPath = Join-Path $repoRoot 'src\audioengine\ISRRuntimeSemanticSchema.h'
$evidenceDir = Join-Path $repoRoot 'evidence'
$reportPath = Join-Path $evidenceDir 'partial_publication_report.json'

if (-not (Test-Path -LiteralPath $evidenceDir)) {
    New-Item -ItemType Directory -Path $evidenceDir -Force | Out-Null
}

$violations = New-Object 'System.Collections.Generic.List[string]'

foreach ($path in @($commitSourcePath, $schemaPath)) {
    if (-not (Test-Path -LiteralPath $path)) {
        $violations.Add("Missing required source file: $path") | Out-Null
    }
}

$commitText = if (Test-Path -LiteralPath $commitSourcePath) {
    Get-Content -LiteralPath $commitSourcePath -Raw -Encoding UTF8
}
else { '' }

$schemaText = if (Test-Path -LiteralPath $schemaPath) {
    Get-Content -LiteralPath $schemaPath -Raw -Encoding UTF8
}
else { '' }

if (-not [regex]::IsMatch($schemaText, '"PartialPublicationVerifier"')) {
    $violations.Add('Required verifier table missing PartialPublicationVerifier entry') | Out-Null
}

if (-not [regex]::IsMatch($commitText, 'validateSemanticCompleteness\(const RuntimePublishWorld& world\)')) {
    $violations.Add('validateSemanticCompleteness(world) helper must exist in AudioEngine.Commit.cpp') | Out-Null
}

if (-not [regex]::IsMatch($commitText, 'if\s*\(!validateSemanticCompleteness\(world\)\)\s*\r?\n\s*return rejectWithEvidence\(\);')) {
    $violations.Add('runPublicationPrecheckNonRt must reject when validateSemanticCompleteness(world) fails') | Out-Null
}

if (-not [regex]::IsMatch($commitText, 'SemanticTransactionState::Rejected')) {
    $violations.Add('partial publication reject path must transition semantic transaction state to Rejected') | Out-Null
}

$report = [ordered]@{
    schema           = 'partial_publication_report_v1'
    generatedAt      = (Get-Date -Format 'o')
    commitSourcePath = $commitSourcePath
    schemaPath       = $schemaPath
    checks           = [ordered]@{
        partialVerifierEntry     = [regex]::IsMatch($schemaText, '"PartialPublicationVerifier"')
        completenessHelperExists = [regex]::IsMatch($commitText, 'validateSemanticCompleteness\(const RuntimePublishWorld& world\)')
        completenessRejectWired  = [regex]::IsMatch($commitText, 'if\s*\(!validateSemanticCompleteness\(world\)\)\s*\r?\n\s*return rejectWithEvidence\(\);')
        rejectedStatePath        = [regex]::IsMatch($commitText, 'SemanticTransactionState::Rejected')
    }
    violations       = @($violations)
    ready            = ($violations.Count -eq 0)
}

$report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $reportPath -Encoding UTF8
Write-Host "[INFO] report: $reportPath"
if ($violations.Count -gt 0) {
    foreach ($v in $violations) { Write-Host "[ERROR] $v" }
    throw 'partial publication verification failed'
}

Write-Host '[PASS] partial publication verification passed'
