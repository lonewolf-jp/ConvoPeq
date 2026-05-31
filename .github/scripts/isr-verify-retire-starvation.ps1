$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$headerPath = Join-Path $repoRoot 'src\audioengine\AudioEngine.h'
$commitPath = Join-Path $repoRoot 'src\audioengine\AudioEngine.Commit.cpp'
$retireTestsPath = Join-Path $repoRoot 'src\tests\RetireGraceSemanticsTests.cpp'
$evidenceDir = Join-Path $repoRoot 'evidence'
$reportPath = Join-Path $evidenceDir 'retire_starvation_report.json'

if (-not (Test-Path -LiteralPath $evidenceDir)) {
    New-Item -ItemType Directory -Path $evidenceDir -Force | Out-Null
}

$violations = New-Object 'System.Collections.Generic.List[string]'

foreach ($path in @($headerPath, $commitPath, $retireTestsPath)) {
    if (-not (Test-Path -LiteralPath $path)) {
        $violations.Add("Missing retire starvation target: $path") | Out-Null
    }
}

$headerText = if (Test-Path -LiteralPath $headerPath) { Get-Content -LiteralPath $headerPath -Raw -Encoding UTF8 } else { '' }
$commitText = if (Test-Path -LiteralPath $commitPath) { Get-Content -LiteralPath $commitPath -Raw -Encoding UTF8 } else { '' }
$testText = if (Test-Path -LiteralPath $retireTestsPath) { Get-Content -LiteralPath $retireTestsPath -Raw -Encoding UTF8 } else { '' }

if (-not [regex]::IsMatch($headerText, 'maxRetireDeferralEpochs_')) {
    $violations.Add('Retire starvation contract missing maxRetireDeferralEpochs_ field') | Out-Null
}

if (-not [regex]::IsMatch($headerText, 'maxRetireWallClockMs_')) {
    $violations.Add('Retire starvation contract missing maxRetireWallClockMs_ field') | Out-Null
}

if (-not [regex]::IsMatch($commitText, 'hasExceededDeferralThresholds\(')) {
    $violations.Add('Retire starvation contract must evaluate dual-threshold deferral exceedance') | Out-Null
}

if (-not [regex]::IsMatch($commitText, 'canReclaimAfterEscalation\(')) {
    $violations.Add('Retire starvation contract must gate reclaim through escalation safety check') | Out-Null
}

if (-not [regex]::IsMatch($testText, 'testRetireStarvationDualThresholdRules\(')) {
    $violations.Add('Retire starvation dedicated test hook missing (testRetireStarvationDualThresholdRules)') | Out-Null
}

$report = [ordered]@{
    schema          = 'retire_starvation_report_v1'
    generatedAt     = (Get-Date -Format 'o')
    headerPath      = $headerPath
    commitPath      = $commitPath
    retireTestsPath = $retireTestsPath
    checks          = [ordered]@{
        hasMaxRetireDeferralEpochs = [regex]::IsMatch($headerText, 'maxRetireDeferralEpochs_')
        hasMaxRetireWallClockMs    = [regex]::IsMatch($headerText, 'maxRetireWallClockMs_')
        dualThresholdCheck         = [regex]::IsMatch($commitText, 'hasExceededDeferralThresholds\(')
        escalationSafetyCheck      = [regex]::IsMatch($commitText, 'canReclaimAfterEscalation\(')
        dualThresholdTest          = [regex]::IsMatch($testText, 'testRetireStarvationDualThresholdRules\(')
    }
    violations      = @($violations)
    ready           = ($violations.Count -eq 0)
}

$report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $reportPath -Encoding UTF8
Write-Host "[INFO] report: $reportPath"
if ($violations.Count -gt 0) {
    foreach ($v in $violations) { Write-Host "[ERROR] $v" }
    throw 'retire starvation verification failed'
}

Write-Host '[PASS] retire starvation verification passed'
