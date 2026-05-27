$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$evidenceDir = Join-Path $repoRoot 'evidence'
$reportPath = Join-Path $evidenceDir 'ownership_migration_report.json'

if (-not (Test-Path $evidenceDir)) {
    New-Item -Path $evidenceDir -ItemType Directory | Out-Null
}

$headerPath = Join-Path $repoRoot 'src\audioengine\AudioEngine.h'
$lifecyclePath = Join-Path $repoRoot 'src\audioengine\AudioEngine.Processing.DSPCoreLifecycle.cpp'

foreach ($path in @($headerPath, $lifecyclePath)) {
    if (-not (Test-Path $path)) {
        throw "Missing required source file: $path"
    }
}

$headerText = Get-Content -LiteralPath $headerPath -Raw -Encoding UTF8
$lifecycleText = Get-Content -LiteralPath $lifecyclePath -Raw -Encoding UTF8

$violations = New-Object System.Collections.Generic.List[string]

if ($headerText -match 'AudioEngine\* ownerEngine\s*=\s*nullptr;') {
    $violations.Add('DSPCore must not retain an ownerEngine member')
}

if ($headerText -match 'publishRcuEpoch\(\) noexcept \{ return ownerEngine' -or
    $headerText -match 'enterRcuReader\(int tid\) noexcept \{ if \(ownerEngine\)' -or
    $headerText -match 'exitRcuReader\(int tid\) noexcept \{ if \(ownerEngine\)') {
    $violations.Add('DSPCore must not forward RCU reader methods through ownerEngine')
}

if ($lifecycleText -match 'this->ownerEngine\s*=\s*owner;') {
    $violations.Add('DSPCore::prepare must not store the owner engine pointer')
}

if ($lifecycleText -notmatch 'convolverState->prepare\(owner,\s*processingRate,\s*processingBlockSize\);') {
    $violations.Add('DSPCore::prepare must pass owner directly to convolverState->prepare(owner, ...)')
}

$report = [ordered]@{
    schema = 'ownership_migration_report_v1'
    generatedAt = (Get-Date -Format 'o')
    headerPath = $headerPath
    lifecyclePath = $lifecyclePath
    violations = $violations
}

Set-Content -LiteralPath $reportPath -Value ($report | ConvertTo-Json -Depth 6) -Encoding UTF8
Write-Host "[INFO] ownership migration report written: $reportPath"

if ($violations.Count -gt 0) {
    foreach ($violation in $violations) {
        Write-Host "[ERROR] $violation"
    }
    throw "Ownership migration violations detected. count=$($violations.Count)"
}

Write-Host '[PASS] ownership migration gate verified'
