$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$headerPath = Join-Path $repoRoot 'src\audioengine\AudioEngine.h'
$audioRoot = Join-Path $repoRoot 'src\audioengine'
$evidenceDir = Join-Path $repoRoot 'evidence'
$reportPath = Join-Path $evidenceDir 'publication_ownership_report.json'

if (-not (Test-Path -LiteralPath $evidenceDir)) {
    New-Item -ItemType Directory -Path $evidenceDir -Force | Out-Null
}

$violations = @()
$hits = @()

$checks = [ordered]@{
    publishWorldBuilderExists = $false
    publishLifecycleWrapperExists = $false
    retireLifecycleWrapperExists = $false
    coordinatorFactoryExists = $false
    noRtPublicationFactoryCall = $false
}

if (-not (Test-Path -LiteralPath $headerPath)) {
    $violations += "Missing header: $headerPath"
}
else {
    $headerText = Get-Content -LiteralPath $headerPath -Raw -Encoding UTF8

    if ([regex]::IsMatch($headerText, '\bbuildRuntimePublishWorld\s*\(')) { $checks.publishWorldBuilderExists = $true }
    else { $violations += 'Publication ownership contract missing: buildRuntimePublishWorld()' }

    if ([regex]::IsMatch($headerText, '\bonRuntimePublishedNonRt\s*\(') -or
        [regex]::IsMatch($headerText, '\bdidPublishRuntimeNonRt\s*\(')) {
        $checks.publishLifecycleWrapperExists = $true
    }
    else { $violations += 'Publication ownership contract missing: onRuntimePublishedNonRt()/didPublishRuntimeNonRt()' }

    if ([regex]::IsMatch($headerText, '\bonRuntimeRetiredNonRt\s*\(') -or
        [regex]::IsMatch($headerText, '\bwillRetireRuntimeNonRt\s*\(')) {
        $checks.retireLifecycleWrapperExists = $true
    }
    else { $violations += 'Publication ownership contract missing: onRuntimeRetiredNonRt()/willRetireRuntimeNonRt()' }

    if ([regex]::IsMatch($headerText, '\bmakeRuntimePublicationCoordinator\s*\(')) { $checks.coordinatorFactoryExists = $true }
    else { $violations += 'Publication ownership contract missing: makeRuntimePublicationCoordinator()' }
}

$rtFiles = @(
)

$forbiddenRtPatterns = @(
    'makeRuntimePublicationCoordinator\s*\(',
    'commitRuntimePublication\s*\(',
    'retireRuntimePublication\s*\(',
    'onRuntimePublishedNonRt\s*\(',
    'onRuntimeRetiredNonRt\s*\('
)

$rtViolations = 0
foreach ($targetPath in $rtFiles) {
    if (-not (Test-Path -LiteralPath $targetPath)) {
        $violations += "Missing RT file for ownership scan: $targetPath"
        continue
    }

    $text = Get-Content -LiteralPath $targetPath -Raw -Encoding UTF8
    foreach ($pattern in $forbiddenRtPatterns) {
        $m = [regex]::Matches($text, $pattern)
        if ($m.Count -gt 0) {
            $rtViolations += $m.Count
            $hits += [pscustomobject]@{ path = $targetPath; pattern = $pattern; count = $m.Count }
            $violations += "Publication ownership violation: RT path uses publication API path=$targetPath pattern=$pattern count=$($m.Count)"
        }
    }
}

if ($rtViolations -eq 0) {
    $checks.noRtPublicationFactoryCall = $true
}

$report = [ordered]@{
    schema = 'publication_ownership_report_v1'
    generatedAt = (Get-Date -Format 'o')
    headerPath = $headerPath
    checks = $checks
    hits = $hits
    violations = $violations
    ready = ($violations.Count -eq 0)
}

$report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $reportPath -Encoding UTF8
Write-Host "[INFO] report: $reportPath"
if ($violations.Count -gt 0) {
    foreach ($v in $violations) { Write-Host "[ERROR] $v" }
    throw 'publication ownership verification failed'
}

Write-Host '[PASS] publication ownership verification passed'
