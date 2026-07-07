$ErrorActionPreference = 'Stop'
$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$headerPath = Join-Path $repoRoot 'src\audioengine\AudioEngine.h'
$timerPath = Join-Path $repoRoot 'src\audioengine\AudioEngine.Timer.cpp'
$evidenceDir = Join-Path $repoRoot 'evidence'
$reportPath = Join-Path $evidenceDir 'runtime_view_governance_report.json'
if (-not (Test-Path -LiteralPath $evidenceDir)) { New-Item -ItemType Directory -Path $evidenceDir -Force | Out-Null }
$violations = New-Object 'System.Collections.Generic.List[string]'
if (-not (Test-Path -LiteralPath $headerPath)) { $violations.Add("Missing header: $headerPath") | Out-Null } else {
    $t = Get-Content -LiteralPath $headerPath -Raw -Encoding UTF8
    if ($t -notmatch 'struct\s+RuntimeReadView') { $violations.Add('RuntimeReadView definition missing') | Out-Null }
    if ($t -notmatch 'struct\s+RuntimePublishView') { $violations.Add('RuntimePublishView definition missing') | Out-Null }

    $authorityBlockMatch = [regex]::Match($t, 'struct\s+AudioCallbackAuthorityView\s*\{[\s\S]*?\};')
    if (-not $authorityBlockMatch.Success) {
        $violations.Add('AudioCallbackAuthorityView definition missing') | Out-Null
    }
    elseif ([regex]::IsMatch($authorityBlockMatch.Value, 'RuntimeGraph\s*\*')) {
        $violations.Add('AudioCallbackAuthorityView must not retain RuntimeGraph* authority') | Out-Null
    }

    $readViewBlockMatch = [regex]::Match($t, 'struct\s+RuntimeReadView\s*\{[\s\S]*?\};')
    if (-not $readViewBlockMatch.Success) {
        $violations.Add('RuntimeReadView definition block missing') | Out-Null
    }
    elseif ([regex]::IsMatch($readViewBlockMatch.Value, 'RuntimeGraph\s*\*\s+graph')) {
        $violations.Add('RuntimeReadView must not expose RuntimeGraph* directly') | Out-Null
    }

    $publishViewBlockMatch = [regex]::Match($t, 'struct\s+RuntimePublishView\s*\{[\s\S]*?\};')
    if (-not $publishViewBlockMatch.Success) {
        $violations.Add('RuntimePublishView definition block missing') | Out-Null
    }
    elseif ([regex]::IsMatch($publishViewBlockMatch.Value, 'RuntimeGraph\s*\*')) {
        $violations.Add('RuntimePublishView must not retain RuntimeGraph* authority') | Out-Null
    }
}

foreach ($path in @($audioBlockPath, $blockDoublePath, $timerPath)) {
    if (-not (Test-Path -LiteralPath $path)) {
        $violations.Add("Missing runtime-view governance target: $path") | Out-Null
        continue
    }

    $text = Get-Content -LiteralPath $path -Raw -Encoding UTF8
    if ([regex]::IsMatch($text, 'runtimeGraph\s*->')) {
        $violations.Add("Runtime view governance violation: direct runtimeGraph dereference in $path") | Out-Null
    }
    if ([regex]::IsMatch($text, '\bgetRuntimeGraph\s*\(')) {
        $violations.Add("Runtime view governance violation: getRuntimeGraph() usage remains in $path") | Out-Null
    }
}
$report = [ordered]@{ schema = 'runtime_view_governance_report_v1'; generatedAt = (Get-Date -Format 'o'); headerPath = $headerPath; violations = @($violations); ready = ($violations.Count -eq 0) }
$report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $reportPath -Encoding UTF8
Write-Host "[INFO] report: $reportPath"
if ($violations.Count -gt 0) { foreach ($v in $violations) { Write-Host "[ERROR] $v" }; throw 'runtime view governance verification failed' }
Write-Host '[PASS] runtime view governance verification passed'
