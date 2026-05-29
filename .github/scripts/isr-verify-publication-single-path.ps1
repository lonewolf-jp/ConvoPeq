$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$evidenceDir = Join-Path $repoRoot 'evidence'
$reportPath = Join-Path $evidenceDir 'publication_single_path_report.json'

if (-not (Test-Path -LiteralPath $evidenceDir)) {
    New-Item -ItemType Directory -Path $evidenceDir -Force | Out-Null
}

$inventoryScriptPath = Join-Path $repoRoot '.github\scripts\isr-generate-authority-inventory.ps1'
$coordinatorHeaderPath = Join-Path $repoRoot 'src\audioengine\ISRRuntimePublicationCoordinator.h'
$coordinatorSourcePath = Join-Path $repoRoot 'src\audioengine\ISRRuntimePublicationCoordinator.cpp'
$commitSourcePath = Join-Path $repoRoot 'src\audioengine\AudioEngine.Commit.cpp'
$audioEngineSourceRoot = Join-Path $repoRoot 'src\audioengine'

foreach ($path in @($inventoryScriptPath, $coordinatorHeaderPath, $coordinatorSourcePath, $commitSourcePath)) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Publication single-path gate missing required file: $path"
    }
}

$inventoryScript = Get-Content -LiteralPath $inventoryScriptPath -Raw -Encoding UTF8
$coordinatorHeader = Get-Content -LiteralPath $coordinatorHeaderPath -Raw -Encoding UTF8
$coordinatorSource = Get-Content -LiteralPath $coordinatorSourcePath -Raw -Encoding UTF8
$commitSource = Get-Content -LiteralPath $commitSourcePath -Raw -Encoding UTF8

$violations = New-Object 'System.Collections.Generic.List[string]'

if (-not $inventoryScript.Contains("publication_path = 'publish(RuntimeWorld*)'")) {
    $violations.Add("Inventory generator publication_path contract mismatch: expected publish(RuntimeWorld*)")
}

$headerCommitCount = ([regex]::Matches($coordinatorHeader, 'void\s+commit\s*\(')).Count
if ($headerCommitCount -ne 1) {
    $violations.Add("RuntimePublicationCoordinator commit declaration count mismatch: expected=1 actual=$headerCommitCount")
}

$sourceCommitCount = ([regex]::Matches($coordinatorSource, 'RuntimePublicationCoordinator::commit\s*\(')).Count
if ($sourceCommitCount -ne 1) {
    $violations.Add("RuntimePublicationCoordinator commit definition count mismatch: expected=1 actual=$sourceCommitCount")
}

$commitRuntimePublicationCount = ([regex]::Matches($commitSource, 'commitRuntimePublication\(world\)')).Count
if ($commitRuntimePublicationCount -ne 1) {
    $violations.Add("publish-to-commit call count mismatch: expected=1 actual=$commitRuntimePublicationCount")
}

if (-not [regex]::IsMatch($commitSource, 'runPublicationPrecheckNonRt\(const RuntimePublishWorld& world\)')) {
    $violations.Add('runPublicationPrecheckNonRt signature must keep const RuntimePublishWorld& contract')
}

if (-not [regex]::IsMatch($commitSource, 'world\.isFrozen\s*\(\s*\)')) {
    $violations.Add('publish precheck must require world.isFrozen() before commit')
}

if (-not [regex]::IsMatch($commitSource, 'world\.isSealedRecursively\s*\(\s*\)')) {
    $violations.Add('publish precheck must require world.isSealedRecursively() before commit')
}

$forbiddenPostPublishMutationPatterns = @(
    'const_cast\s*<\s*RuntimePublishWorld\s*\*\s*>',
    'world\s*\.\s*unseal\s*\(',
    'world\s*\.\s*seal\s*\(',
    'world\s*\.\s*sealRecursively\s*\(',
    'world\s*\.\s*freeze\s*\(',
    'world\s*\.\s*assertMutable\s*\('
)

foreach ($pattern in $forbiddenPostPublishMutationPatterns) {
    if ([regex]::IsMatch($commitSource, $pattern)) {
        $violations.Add("Forbidden post-publish mutation pattern detected in AudioEngine.Commit.cpp: $pattern")
    }
}

$runtimeStoreDirectAccessHits = New-Object 'System.Collections.Generic.List[string]'
$audioEngineSourceFiles = Get-ChildItem -LiteralPath $audioEngineSourceRoot -Recurse -File -Include '*.h','*.cpp'
foreach ($sourceFile in $audioEngineSourceFiles) {
    $text = Get-Content -LiteralPath $sourceFile.FullName -Raw -Encoding UTF8
    $matches = [regex]::Matches($text, 'runtimeStore\s*\.')
    if ($matches.Count -gt 0) {
        $relative = [System.IO.Path]::GetRelativePath($repoRoot, $sourceFile.FullName).Replace('\\', '/')
        $runtimeStoreDirectAccessHits.Add($relative) | Out-Null
    }
}

if ($runtimeStoreDirectAccessHits.Count -gt 0) {
    foreach ($hit in $runtimeStoreDirectAccessHits) {
        $violations.Add("Forbidden direct runtimeStore member access detected: $hit")
    }
}

$forbiddenPublicationApis = @('publishGraph', 'publishTransition', 'publishSnapshot', 'publishFade')
foreach ($api in $forbiddenPublicationApis) {
    if ($coordinatorSource.Contains($api) -or $coordinatorHeader.Contains($api)) {
        $violations.Add("Forbidden field-level publication API detected: $api")
    }
}

$report = [ordered]@{
    schema = 'publication_single_path_report_v1'
    generatedAt = (Get-Date -Format 'o')
    files = [ordered]@{
        inventoryScript = $inventoryScriptPath
        coordinatorHeader = $coordinatorHeaderPath
        coordinatorSource = $coordinatorSourcePath
        commitSource = $commitSourcePath
    }
    checks = [ordered]@{
        publicationPathContract = $inventoryScript.Contains("publication_path = 'publish(RuntimeWorld*)'")
        commitDeclarationCount = $headerCommitCount
        commitDefinitionCount = $sourceCommitCount
        commitRuntimePublicationCount = $commitRuntimePublicationCount
        frozenPrecheck = [regex]::IsMatch($commitSource, 'world\.isFrozen\s*\(\s*\)')
        sealedPrecheck = [regex]::IsMatch($commitSource, 'world\.isSealedRecursively\s*\(\s*\)')
        forbiddenPostPublishMutationPatterns = @($forbiddenPostPublishMutationPatterns)
        runtimeStoreDirectAccessHits = @($runtimeStoreDirectAccessHits)
        forbiddenApiScan = @($forbiddenPublicationApis)
    }
    violations = @($violations)
    ready = ($violations.Count -eq 0)
}

$report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $reportPath -Encoding UTF8
Write-Host "[INFO] publication single-path report written: $reportPath"

if ($violations.Count -gt 0) {
    foreach ($violation in $violations) {
        Write-Host "[ERROR] $violation"
    }
    throw "Publication single-path verification failed. violations=$($violations.Count)"
}

Write-Host '[PASS] Publication single-path verification passed'
