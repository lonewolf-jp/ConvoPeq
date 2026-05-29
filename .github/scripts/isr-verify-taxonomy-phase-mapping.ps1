$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$work5Dir = Join-Path $repoRoot 'doc\work5'
$evidenceDir = Join-Path $repoRoot 'evidence'
$reportPath = Join-Path $evidenceDir 'taxonomy_phase_mapping_report.json'

if (-not (Test-Path -LiteralPath $work5Dir)) {
    throw "Missing work5 directory: $work5Dir"
}

if (-not (Test-Path -LiteralPath $evidenceDir)) {
    New-Item -Path $evidenceDir -ItemType Directory | Out-Null
}

$candidates = @(Get-ChildItem -LiteralPath $work5Dir -File -Filter 'Practical_Stable_ISR_Runtime_*_v1_2.md' | Sort-Object -Property Name)
$designDoc = $null
$designText = ''

foreach ($candidate in $candidates) {
    $text = Get-Content -LiteralPath $candidate.FullName -Raw -Encoding UTF8
    if ($text.Contains('Soak Failure Taxonomy') -and $text.Contains('Phase DoD')) {
        $designDoc = $candidate
        $designText = $text
        break
    }
}

if ($null -eq $designDoc) {
    throw 'Failed to resolve design v1.2 document with taxonomy/DoD section'
}

$violations = New-Object 'System.Collections.Generic.List[string]'

$requiredMappings = @(
    [ordered]@{ cls = 'Class-A'; phase = 'Phase 4, 5'; lineRegex = '\|\s*Class-A\s+audio corruption\s*\|[\s\S]*?\|\s*Phase 4, 5\s*\|' },
    [ordered]@{ cls = 'Class-B'; phase = 'Phase 1, 5'; lineRegex = '\|\s*Class-B\s+generation drift\s*\|[\s\S]*?\|\s*Phase 1, 5\s*\|' },
    [ordered]@{ cls = 'Class-C'; phase = 'Phase 2, 5'; lineRegex = '\|\s*Class-C\s+stale observe\s*\|[\s\S]*?\|\s*Phase 2, 5\s*\|' },
    [ordered]@{ cls = 'Class-D'; phase = 'Phase 6'; lineRegex = '\|\s*Class-D\s+backlog divergence\s*\|[\s\S]*?\|\s*Phase 6\s*\|' },
    [ordered]@{ cls = 'Class-E'; phase = 'Phase 6'; lineRegex = '\|\s*Class-E\s+retention leak\s*\|[\s\S]*?\|\s*Phase 6\s*\|' },
    [ordered]@{ cls = 'Class-F'; phase = 'Phase 1, 3'; lineRegex = '\|\s*Class-F\s+authority duplication\s*\|[\s\S]*?\|\s*Phase 1, 3\s*\|' }
)

foreach ($mapping in $requiredMappings) {
    if (-not [regex]::IsMatch($designText, $mapping.lineRegex)) {
        $violations.Add("Missing taxonomy-phase mapping: class=$($mapping.cls) expectedPhase=$($mapping.phase)")
    }
}

$requiredActions = @(
    'Class-A/B/C:.*fail',
    'Class-D/E:.*fail',
    'Class-F:.*fail'
)

foreach ($actionPattern in $requiredActions) {
    if (-not [regex]::IsMatch($designText, $actionPattern)) {
        $violations.Add("Missing taxonomy action contract: pattern=$actionPattern")
    }
}

$report = [ordered]@{
    schema = 'taxonomy_phase_mapping_report_v1'
    generatedAt = (Get-Date -Format 'o')
    sourceDocument = $designDoc.FullName
    checks = [ordered]@{
        mappingRows = ($violations | Where-Object { $_ -like 'Missing taxonomy-phase mapping*' }).Count -eq 0
        actionRules = ($violations | Where-Object { $_ -like 'Missing taxonomy action contract*' }).Count -eq 0
    }
    violations = @($violations)
    ready = ($violations.Count -eq 0)
}

$report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $reportPath -Encoding UTF8
Write-Host "[INFO] taxonomy phase mapping report written: $reportPath"

if ($violations.Count -gt 0) {
    foreach ($v in $violations) {
        Write-Host "[ERROR] $v"
    }
    throw "Taxonomy phase mapping verification failed. violations=$($violations.Count)"
}

Write-Host '[PASS] Taxonomy phase mapping verification passed'
