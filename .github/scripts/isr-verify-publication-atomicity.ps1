$ErrorActionPreference = 'Stop'
$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$commitPath = Join-Path $repoRoot 'src\audioengine\AudioEngine.Commit.cpp'
$reportPath = Join-Path $repoRoot 'evidence\publication_atomicity_report.json'
if (-not(Test-Path -LiteralPath (Split-Path $reportPath -Parent))) { New-Item -ItemType Directory -Path (Split-Path $reportPath -Parent) -Force | Out-Null }
$violations = New-Object 'System.Collections.Generic.List[string]'
if (-not(Test-Path -LiteralPath $commitPath)) {
    $violations.Add("Missing source: $commitPath") | Out-Null
}
else {
    $s = Get-Content -LiteralPath $commitPath -Raw -Encoding UTF8

    $requiredPatterns = @(
        'runPublicationPrecheckNonRt\(const RuntimePublishWorld& world\)',
        'if \(world\.publication\.previousSequenceId >= world\.publication\.sequenceId\)',
        'if \(lastCommittedGeneration != 0 && world\.generation <= lastCommittedGeneration\)',
        'if \(lastCommittedSequence != 0 && world\.publication\.sequenceId <= lastCommittedSequence\)',
        'if \(targetWorldIdU64 <= lastEnqueuedTargetWorldId\)',
        'publishAtomic\(lastEnqueuedPublicationTargetWorldId_',
        'commitRuntimePublication\(world\)',
        'publishAtomic\(lastCommittedRuntimeGeneration_',
        'publishAtomic\(lastCommittedPublicationSequence_'
    )

    foreach ($pattern in $requiredPatterns) {
        if ($s -notmatch $pattern) {
            $violations.Add("Missing publication atomicity contract pattern: $pattern") | Out-Null
        }
    }

    $commitIndex = $s.IndexOf('commitRuntimePublication(world);', [System.StringComparison]::Ordinal)
    $publishGenerationIndex = $s.IndexOf('publishAtomic(lastCommittedRuntimeGeneration_', [System.StringComparison]::Ordinal)
    $publishSequenceIndex = $s.IndexOf('publishAtomic(lastCommittedPublicationSequence_', [System.StringComparison]::Ordinal)

    if ($commitIndex -lt 0) {
        $violations.Add('Atomic publication commit call not found') | Out-Null
    }
    else {
        if ($publishGenerationIndex -lt 0 -or $publishGenerationIndex -le $commitIndex) {
            $violations.Add('Atomic publication ordering violation: lastCommittedRuntimeGeneration_ must publish after commitRuntimePublication(world)') | Out-Null
        }

        if ($publishSequenceIndex -lt 0 -or $publishSequenceIndex -le $commitIndex) {
            $violations.Add('Atomic publication ordering violation: lastCommittedPublicationSequence_ must publish after commitRuntimePublication(world)') | Out-Null
        }
    }
}
$report = [ordered]@{schema = 'publication_atomicity_report_v1'; generatedAt = (Get-Date -Format 'o'); sourcePath = $commitPath; violations = @($violations); ready = ($violations.Count -eq 0) }
$report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $reportPath -Encoding UTF8
Write-Host "[INFO] report: $reportPath"; if ($violations.Count -gt 0) { foreach ($v in $violations) { Write-Host "[ERROR] $v" }; throw 'publication atomicity verification failed' }
Write-Host '[PASS] publication atomicity verification passed'
