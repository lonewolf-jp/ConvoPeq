$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$evidenceDir = Join-Path $repoRoot 'evidence'
$reportPath = Join-Path $evidenceDir 'audio_startup_order_report.json'

if (-not (Test-Path -LiteralPath $evidenceDir)) {
    New-Item -Path $evidenceDir -ItemType Directory | Out-Null
}

$targets = @(
)

$violations = @()
$checks = @()

foreach ($target in $targets) {
    $targetPath = [string]$target.path
    if (-not (Test-Path -LiteralPath $targetPath)) {
        $violations += "Missing target source file: $targetPath"
        continue
    }

    $content = Get-Content -LiteralPath $targetPath -Raw -Encoding UTF8
    $functionSignature = [string]$target.functionSignature
    $sigIndex = $content.IndexOf($functionSignature, [System.StringComparison]::Ordinal)
    if ($sigIndex -lt 0) {
        $violations += "Missing function signature in target: $functionSignature"
        continue
    }

    $windowLength = [Math]::Min(1200, $content.Length - $sigIndex)
    if ($windowLength -le 0) {
        $violations += "Unable to scan function window: $functionSignature"
        continue
    }

    $window = $content.Substring($sigIndex, $windowLength)

    $hasLifecycleRead = $window.Contains('consumeAtomic(lifecycleState')
    $hasPreparedGuard = $window.Contains('EngineLifecycleState::Prepared')
    $hasClearOnGuard = $window.Contains([string]$target.clearExpression)
    $usesHandleRuntime = $window.Contains('dspHandleRuntime_')

    if (-not $hasLifecycleRead -or -not $hasPreparedGuard -or -not $hasClearOnGuard) {
        $violations += "Audio startup order guard missing or incomplete: file=$targetPath lifecycleRead=$hasLifecycleRead preparedGuard=$hasPreparedGuard clearOnGuard=$hasClearOnGuard"
    }

    if ($usesHandleRuntime) {
        $violations += "Audio callback must observe RuntimeWorld only (dspHandleRuntime_ is forbidden): file=$targetPath"
    }

    $checks += [ordered]@{
        path = $targetPath
        functionSignature = $functionSignature
        hasLifecycleRead = $hasLifecycleRead
        hasPreparedGuard = $hasPreparedGuard
        hasClearOnGuard = $hasClearOnGuard
        usesHandleRuntime = $usesHandleRuntime
    }
}

$report = [ordered]@{
    schema = 'audio_startup_order_report_v1'
    generatedAt = (Get-Date -Format 'o')
    checks = @($checks)
    violations = @($violations)
    ready = ($violations.Count -eq 0)
}

Set-Content -LiteralPath $reportPath -Value ($report | ConvertTo-Json -Depth 6) -Encoding UTF8
Write-Host "[INFO] audio startup order report written: $reportPath"

if ($violations.Count -gt 0) {
    foreach ($violation in $violations) {
        Write-Host "[ERROR] $violation"
    }
    throw "Audio startup order verification failed. violations=$($violations.Count)"
}

Write-Host '[PASS] audio startup order gate verified'
