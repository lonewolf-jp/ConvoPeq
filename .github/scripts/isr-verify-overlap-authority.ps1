$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$headerPath = Join-Path $repoRoot 'src\audioengine\AudioEngine.h'
$evidenceDir = Join-Path $repoRoot 'evidence'
$reportPath = Join-Path $evidenceDir 'overlap_authority_report.json'

if (-not (Test-Path -LiteralPath $evidenceDir)) {
    New-Item -ItemType Directory -Path $evidenceDir -Force | Out-Null
}

$violations = New-Object 'System.Collections.Generic.List[string]'

foreach ($path in @($headerPath, $audioBlockPath, $blockDoublePath)) {
    if (-not (Test-Path -LiteralPath $path)) {
        $violations.Add("Missing overlap authority target: $path") | Out-Null
    }
}

if (Test-Path -LiteralPath $headerPath) {
    $headerText = Get-Content -LiteralPath $headerPath -Raw -Encoding UTF8

    if (-not [regex]::IsMatch($headerText, 'makeCrossfadePreparedSnapshotFromWorld\s*\(')) {
        $violations.Add('Overlap authority helper missing: makeCrossfadePreparedSnapshotFromWorld') | Out-Null
    }

    if (-not [regex]::IsMatch($headerText, 'struct\s+AudioCallbackAuthorityView\s*\{[\s\S]*preparedCrossfade[\s\S]*\};')) {
        $violations.Add('AudioCallbackAuthorityView must expose preparedCrossfade sourced from RuntimeWorld') | Out-Null
    }
}

foreach ($path in @($audioBlockPath, $blockDoublePath)) {
    if (-not (Test-Path -LiteralPath $path)) {
        continue
    }

    $text = Get-Content -LiteralPath $path -Raw -Encoding UTF8

    if (-not [regex]::IsMatch($text, 'AudioCallbackAuthorityView\s*\{\s*makeCrossfadePreparedSnapshotFromWorld\s*\(')) {
        $violations.Add("Overlap authority violation: missing RuntimeWorld overlap snapshot bridge in $path") | Out-Null
    }

    if (-not [regex]::IsMatch($text, 'authority\.preparedCrossfade')) {
        $violations.Add("Overlap authority violation: callback must read overlap branch from authority.preparedCrossfade in $path") | Out-Null
    }

    if ([regex]::IsMatch($text, '\bcrossfadePreparedSnapshot\b')) {
        $violations.Add("Overlap authority violation: legacy crossfadePreparedSnapshot authority token detected in $path") | Out-Null
    }
}

$report = [ordered]@{
    schema          = 'overlap_authority_report_v1'
    generatedAt     = (Get-Date -Format 'o')
    headerPath      = $headerPath
    audioBlockPath  = $audioBlockPath
    blockDoublePath = $blockDoublePath
    violations      = @($violations)
    ready           = ($violations.Count -eq 0)
}

$report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $reportPath -Encoding UTF8
Write-Host "[INFO] report: $reportPath"
if ($violations.Count -gt 0) {
    foreach ($v in $violations) {
        Write-Host "[ERROR] $v"
    }
    throw 'overlap authority verification failed'
}

Write-Host '[PASS] overlap authority verification passed'
