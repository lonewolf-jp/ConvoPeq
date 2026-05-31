$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$audioBlockPath = Join-Path $repoRoot 'src\audioengine\AudioEngine.Processing.AudioBlock.cpp'
$blockDoublePath = Join-Path $repoRoot 'src\audioengine\AudioEngine.Processing.BlockDouble.cpp'
$evidenceDir = Join-Path $repoRoot 'evidence'
$reportPath = Join-Path $evidenceDir 'observe_path_report.json'

if (-not (Test-Path -LiteralPath $evidenceDir)) {
    New-Item -ItemType Directory -Path $evidenceDir -Force | Out-Null
}

$violations = New-Object 'System.Collections.Generic.List[string]'

foreach ($path in @($audioBlockPath, $blockDoublePath)) {
    if (-not (Test-Path -LiteralPath $path)) {
        $violations.Add("Missing observe path target: $path") | Out-Null
        continue
    }

    $text = Get-Content -LiteralPath $path -Raw -Encoding UTF8

    if (-not [regex]::IsMatch($text, 'const auto\* runtimeWorld = runtimeReadViewRef\.runtimeWorld;')) {
        $violations.Add("ObservePathVerifier: runtimeWorld read-view bridge missing in $path") | Out-Null
    }

    if (-not [regex]::IsMatch($text, 'if \(runtimeWorld == nullptr\)')) {
        $violations.Add("ObservePathVerifier: runtimeWorld null guard missing in $path") | Out-Null
    }

    if (-not [regex]::IsMatch($text, 'makeCrossfadePreparedSnapshotFromWorld\(\*runtimeWorld\)')) {
        $violations.Add("ObservePathVerifier: world-derived authority snapshot missing in $path") | Out-Null
    }

    if ([regex]::IsMatch($text, '\bgetRuntimeGraph\s*\(')) {
        $violations.Add("ObservePathVerifier: forbidden getRuntimeGraph() observe path detected in $path") | Out-Null
    }

    if ([regex]::IsMatch($text, 'runtimeGraph\s*->')) {
        $violations.Add("ObservePathVerifier: forbidden runtimeGraph-> dereference detected in $path") | Out-Null
    }
}

$report = [ordered]@{
    schema      = 'observe_path_report_v1'
    generatedAt = (Get-Date -Format 'o')
    targets     = @($audioBlockPath, $blockDoublePath)
    violations  = @($violations)
    ready       = ($violations.Count -eq 0)
}

$report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $reportPath -Encoding UTF8
Write-Host "[INFO] report: $reportPath"
if ($violations.Count -gt 0) {
    foreach ($v in $violations) { Write-Host "[ERROR] $v" }
    throw 'observe path verification failed'
}

Write-Host '[PASS] observe path verification passed'
