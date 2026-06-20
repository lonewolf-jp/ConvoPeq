$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$schemaPath = Join-Path $repoRoot 'src\audioengine\ISRRuntimeSemanticSchema.h'
$headerPath = Join-Path $repoRoot 'src\audioengine\AudioEngine.h'
$builderPath = Join-Path $repoRoot 'src\audioengine\RuntimeBuilder.cpp'
$commitPath = Join-Path $repoRoot 'src\audioengine\AudioEngine.Commit.cpp'
$coordinatorPath = Join-Path $repoRoot 'src\audioengine\ISRRuntimePublicationCoordinator.cpp'
$coordinatorHeaderPath = Join-Path $repoRoot 'src\audioengine\ISRRuntimePublicationCoordinator.h'
$evidenceDir = Join-Path $repoRoot 'evidence'
$reportPath = Join-Path $evidenceDir 'aba_hazard_report.json'

if (-not (Test-Path -LiteralPath $evidenceDir)) {
    New-Item -ItemType Directory -Path $evidenceDir -Force | Out-Null
}

$violations = New-Object 'System.Collections.Generic.List[string]'

foreach ($path in @($schemaPath, $headerPath, $builderPath, $commitPath, $coordinatorPath)) {
    if (-not (Test-Path -LiteralPath $path)) {
        $violations.Add("Missing required file: $path") | Out-Null
    }
}

$schemaText = if (Test-Path -LiteralPath $schemaPath) { Get-Content -LiteralPath $schemaPath -Raw -Encoding UTF8 } else { '' }
$headerText = if (Test-Path -LiteralPath $headerPath) { Get-Content -LiteralPath $headerPath -Raw -Encoding UTF8 } else { '' }
$builderText = if (Test-Path -LiteralPath $builderPath) { Get-Content -LiteralPath $builderPath -Raw -Encoding UTF8 } else { '' }
$commitText = if (Test-Path -LiteralPath $commitPath) { Get-Content -LiteralPath $commitPath -Raw -Encoding UTF8 } else { '' }
$coordinatorText = if (Test-Path -LiteralPath $coordinatorPath) { Get-Content -LiteralPath $coordinatorPath -Raw -Encoding UTF8 } else { '' }
$coordinatorHeaderText = if (Test-Path -LiteralPath $coordinatorHeaderPath) { Get-Content -LiteralPath $coordinatorHeaderPath -Raw -Encoding UTF8 } else { '' }
# isMonotonic と内部実装はヘッダのインライン定義のため両方を結合
$coordinatorCombinedText = $coordinatorText + "`n" + $coordinatorHeaderText

if (-not $schemaText.Contains('"ABAHazardVerifier"')) {
    $violations.Add('kRequiredVerifierTable must register ABAHazardVerifier') | Out-Null
}

$headerPatterns = @(
    'RuntimeWorldIdGenerator runtimeWorldIdGenerator_'
)

$builderPatterns = @(
    'worldOwner->worldId = nextWorldId;',
    'worldOwner->generation = nextGraphGeneration;',
    'worldOwner->retire.retireEpoch = nextGraphGeneration;',
    'worldOwner->publication.sequenceId = nextPublicationSequence;',
    'worldOwner->publication.mappedRuntimeGeneration = nextGraphGeneration;'
)

foreach ($pattern in $headerPatterns) {
    if (-not [regex]::IsMatch($headerText, [regex]::Escape($pattern))) {
        $violations.Add("ABA identity contract pattern missing in AudioEngine.h: $pattern") | Out-Null
    }
}

foreach ($pattern in $builderPatterns) {
    if (-not [regex]::IsMatch($builderText, [regex]::Escape($pattern))) {
        $violations.Add("ABA identity contract pattern missing in RuntimeBuilder.cpp: $pattern") | Out-Null
    }
}

# 以下の commitPatterns は過去のリファクタリングで削除済みのためチェックしない:
# - targetWorldIdU64 / lastEnqueuedTargetWorldId (PublicationIntent 系)
# - publicationIntentRequestIdCounter_

$coordinatorPatterns = @(
    'PersistentStateBlock::isMonotonic\(prev,',
    'nextSeqId > prev\.publicationSequenceId',
    'nextEpoch > prev\.publicationEpoch',
    'nextGen > prev\.mappedRuntimeGeneration',
    'persistentState_\s*=\s*PersistentStateBlock\{'
)

foreach ($pattern in $coordinatorPatterns) {
    if (-not [regex]::IsMatch($coordinatorCombinedText, $pattern, [System.Text.RegularExpressions.RegexOptions]::Singleline)) {
        $violations.Add("ABA contract pattern missing in ISRRuntimePublicationCoordinator: $pattern") | Out-Null
    }
}

$report = [ordered]@{
    schema = 'aba_hazard_report_v1'
    generatedAt = (Get-Date -Format 'o')
    headerPath = $headerPath
    builderPath = $builderPath
    commitPath = $commitPath
    coordinatorPath = $coordinatorPath
    violations = @($violations)
    ready = ($violations.Count -eq 0)
}

$report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $reportPath -Encoding UTF8
Write-Host "[INFO] report: $reportPath"
if ($violations.Count -gt 0) {
    foreach ($v in $violations) { Write-Host "[ERROR] $v" }
    throw 'ABA hazard verification failed'
}

Write-Host '[PASS] ABA hazard verification passed'
