param(
    [string]$RepoRoot = (Join-Path $PSScriptRoot '..\..')
)

$ErrorActionPreference = 'Stop'

$repoRootResolved = [System.IO.Path]::GetFullPath($RepoRoot)
$evidenceDir = Join-Path $repoRootResolved 'evidence'
if (-not (Test-Path -LiteralPath $evidenceDir)) {
    New-Item -ItemType Directory -Path $evidenceDir -Force | Out-Null
}

$reportPath = Join-Path $evidenceDir 'c1_c15_minimal_report.json'

$prtRoots = @(
    (Join-Path $repoRootResolved 'src\audioengine'),
    (Join-Path $repoRootResolved 'src\convolver'),
    (Join-Path $repoRootResolved 'src\eqprocessor'),
    (Join-Path $repoRootResolved 'src\core')
)

function Get-SourceFiles {
    param([string[]]$Roots)

    $files = New-Object System.Collections.Generic.List[System.IO.FileInfo]
    foreach ($root in $Roots) {
        if (Test-Path -LiteralPath $root) {
            Get-ChildItem -LiteralPath $root -Recurse -File -Include *.h, *.hpp, *.cpp, *.cxx, *.cc | ForEach-Object {
                $files.Add($_) | Out-Null
            }
        }
    }

    return @($files)
}

function Count-RegexMatches {
    param(
        [System.IO.FileInfo[]]$Files,
        [string]$Pattern
    )

    $count = 0
    $regex = [regex]::new($Pattern)
    foreach ($file in $Files) {
        $text = Get-Content -LiteralPath $file.FullName -Raw -Encoding UTF8
        $count += $regex.Matches($text).Count
    }

    return $count
}

function New-CheckResult {
    param(
        [string]$Id,
        [string]$Description,
        [string]$Status,
        [string]$Evidence,
        [string]$Note = ''
    )

    return [pscustomobject]@{
        id          = $Id
        description = $Description
        status      = $Status
        evidence    = $Evidence
        note        = $Note
    }
}

$files = Get-SourceFiles -Roots $prtRoots

$results = New-Object System.Collections.Generic.List[object]

# C1/C14: publishState callsite = 0（ISR Bridge Runtime 移行により削除済み）
$publishStateAll = Count-RegexMatches -Files $files -Pattern 'publishState\s*\('
$publishStateDecl = Count-RegexMatches -Files $files -Pattern 'void\s+publishState\s*\('
$publishStateCallsites = [Math]::Max(0, $publishStateAll - $publishStateDecl)
$publishStateStatus = if ($publishStateCallsites -eq 0) { 'pass' } else { 'fail' }
$publishStateEvidence = "publishStateAll=$publishStateAll publishStateDecl=$publishStateDecl callsites=$publishStateCallsites"
$results.Add((New-CheckResult -Id 'C1' -Description 'publishState removed after ISR Bridge migration' -Status $publishStateStatus -Evidence $publishStateEvidence)) | Out-Null
$results.Add((New-CheckResult -Id 'C14' -Description 'publishState() removed after ISR Bridge migration (PRT)' -Status $publishStateStatus -Evidence $publishStateEvidence)) | Out-Null

# C2/C3: legacy publication symbols removed
$c2Count = Count-RegexMatches -Files $files -Pattern '\bcommitRuntimePublication\s*\('
$c3Count = Count-RegexMatches -Files $files -Pattern '\bretireRuntimePublication\s*\('
$c2Status = if ($c2Count -eq 0) { 'pass' } else { 'fail' }
$c3Status = if ($c3Count -eq 0) { 'pass' } else { 'fail' }
$results.Add((New-CheckResult -Id 'C2' -Description 'commitRuntimePublication removed/non-used' -Status $c2Status -Evidence "count=$c2Count")) | Out-Null
$results.Add((New-CheckResult -Id 'C3' -Description 'retireRuntimePublication removed/non-used' -Status $c3Status -Evidence "count=$c3Count")) | Out-Null

# C4: AudioEngine authority legacy ops removed from legacy path
# ISR Bridge 移行後の正規関数は除外（例: commitOrRollbackProbe, publishIdleWorldOnly）
$c4LegacyCommit = Count-RegexMatches -Files $files -Pattern '\bprepareCommit\s*\('
$c4LegacyExecute = Count-RegexMatches -Files $files -Pattern '\bexecuteCommit\s*\('
$c4LegacyCommitNewDsp = Count-RegexMatches -Files $files -Pattern '\bcommitNewDSP\s*\('
$c4BridgeCommit = Count-RegexMatches -Files $files -Pattern 'runtimePublicationBridge_\.commit\s*\('
$c4BridgeRetire = Count-RegexMatches -Files $files -Pattern 'runtimePublicationBridge_\.retire\s*\('
$c4ForbiddenAudioEngineOps = Count-RegexMatches -Files $files -Pattern '\bAudioEngine::(?:commit|publish|retire|build|activate)\w*\s*\('
# ISR Bridge 正規関数を除外（commitOrRollbackProbe, publishIdleWorldOnly）
$c4IsrBridgeOps = Count-RegexMatches -Files $files -Pattern '\bAudioEngine::(?:commitOrRollbackProbe|publishIdleWorldOnly)\s*\('
$c4ForbiddenAudioEngineOps = [Math]::Max(0, $c4ForbiddenAudioEngineOps - $c4IsrBridgeOps)
$c4Status = if ($c4LegacyCommit -eq 0 -and $c4LegacyExecute -eq 0 -and $c4LegacyCommitNewDsp -eq 0 -and $c4BridgeCommit -ge 1 -and $c4BridgeRetire -ge 1 -and $c4ForbiddenAudioEngineOps -eq 0) { 'pass' } else { 'fail' }
$c4Evidence = "legacyPrepareCommit=$c4LegacyCommit legacyExecuteCommit=$c4LegacyExecute legacyCommitNewDSP=$c4LegacyCommitNewDsp bridgeCommit=$c4BridgeCommit bridgeRetire=$c4BridgeRetire forbiddenAudioEngineOps=$c4ForbiddenAudioEngineOps"
$results.Add((New-CheckResult -Id 'C4' -Description 'AudioEngine authority operations removed from legacy path' -Status $c4Status -Evidence $c4Evidence)) | Out-Null

# C6: execution branch must not depend on transition.active / execution.transitionActive
$c6ExecutionPaths = @(
    (Join-Path $repoRootResolved 'src\audioengine\AudioEngine.Processing.Snapshot.cpp'),
    (Join-Path $repoRootResolved 'src\audioengine\AudioEngine.Timer.cpp')
)
$c6ExecutionFiles = @($c6ExecutionPaths | Where-Object { Test-Path -LiteralPath $_ } | ForEach-Object { Get-Item -LiteralPath $_ })
$c6TokenCount = Count-RegexMatches -Files $c6ExecutionFiles -Pattern '\b(?:transition\.active|execution\.transitionActive)\b'
$c6Status = if ($c6TokenCount -eq 0) { 'pass' } else { 'fail' }
$results.Add((New-CheckResult -Id 'C6' -Description 'Execution branch does not depend on transition.active aliases' -Status $c6Status -Evidence "tokenCount=$c6TokenCount files=$($c6ExecutionFiles.Count)")) | Out-Null

# C7: generation authority singularity（minimal heuristic）
$audioEngineH = Join-Path $repoRootResolved 'src\audioengine\AudioEngine.h'
$schemaH = Join-Path $repoRootResolved 'src\audioengine\ISRRuntimeSemanticSchema.h'
$c7Status = 'fail'
$c7Evidence = 'missing files'
if ((Test-Path -LiteralPath $audioEngineH) -and (Test-Path -LiteralPath $schemaH)) {
    $audioText = Get-Content -LiteralPath $audioEngineH -Raw -Encoding UTF8
    $schemaText = Get-Content -LiteralPath $schemaH -Raw -Encoding UTF8

    $hasGenerationAuthoritative = ($audioText -match '\{"generation",\s*convo::isr::RuntimeAuthorityClass::Authoritative\}')
    $hasGenerationSemanticDerived = ($audioText -match '\{"generationSemantic",\s*convo::isr::RuntimeAuthorityClass::Derived\}')
    $hasMappedDerived = ($schemaText -match '\{"mappedRuntimeGeneration",\s*SemanticCategory::Derived')

    if ($hasGenerationAuthoritative -and $hasGenerationSemanticDerived -and $hasMappedDerived) {
        $c7Status = 'pass'
    }

    $c7Evidence = "generationAuthoritative=$hasGenerationAuthoritative generationSemanticDerived=$hasGenerationSemanticDerived mappedRuntimeGenerationDerived=$hasMappedDerived"
}
$results.Add((New-CheckResult -Id 'C7' -Description 'Generation authority = 1 (minimal heuristic)' -Status $c7Status -Evidence $c7Evidence)) | Out-Null

# C5: RuntimeState self-contained execution metadata exists
$c5Status = 'fail'
$c5Evidence = 'missing files'
if ((Test-Path -LiteralPath $audioEngineH) -and (Test-Path -LiteralPath $schemaH)) {
    $audioText = Get-Content -LiteralPath $audioEngineH -Raw -Encoding UTF8

    $hasRuntimeState = ($audioText -match 'struct\s+RuntimeState')
    $hasExecutionSemantic = ($audioText -match 'convo::isr::ExecutionSemantic\s+execution\s*\{\s*\};')
    $hasPublicationSemantic = ($audioText -match 'convo::isr::PublicationSemantic\s+publication\s*\{\s*\};')
    $hasRetireSemantic = ($audioText -match 'convo::isr::RetireSemantic\s+retire\s*\{\s*\};')
    $hasDescriptorValidation = ($audioText -match 'validateDescriptorSet\(\)\s+noexcept')

    if ($hasRuntimeState -and $hasExecutionSemantic -and $hasPublicationSemantic -and $hasRetireSemantic -and $hasDescriptorValidation) {
        $c5Status = 'pass'
    }

    $c5Evidence = "runtimeState=$hasRuntimeState executionSemantic=$hasExecutionSemantic publicationSemantic=$hasPublicationSemantic retireSemantic=$hasRetireSemantic descriptorValidation=$hasDescriptorValidation"
}
$results.Add((New-CheckResult -Id 'C5' -Description 'RuntimeState self-contained execution metadata present' -Status $c5Status -Evidence $c5Evidence)) | Out-Null

# C8: delegate to retire authority verifier
$c8Status = 'fail'
$c8Evidence = ''
try {
    & (Join-Path $PSScriptRoot 'isr-verify-v5-retire-authority-lane.ps1') | Out-Null
    $audioEngineRoot = Join-Path $repoRootResolved 'src\audioengine'
    $audioEngineFiles = if (Test-Path -LiteralPath $audioEngineRoot) {
        Get-ChildItem -LiteralPath $audioEngineRoot -Recurse -File -Include *.h, *.hpp, *.cpp, *.cxx, *.cc
    }
    else {
        @()
    }

    $audioEngineFallbackQueueCount = Count-RegexMatches -Files $audioEngineFiles -Pattern '\b(?:deferredDeleteFallbackQueue|deferredRetireFallbackQueue_)\b'
    $c8Status = if ($audioEngineFallbackQueueCount -eq 0) { 'pass' } else { 'fail' }
    $c8Evidence = "isr-verify-v5-retire-authority-lane.ps1=PASS audioEngineFallbackQueueCount=$audioEngineFallbackQueueCount"
}
catch {
    $c8Status = 'fail'
    $c8Evidence = "isr-verify-v5-retire-authority-lane.ps1 failed: $($_.Exception.Message)"
}
$results.Add((New-CheckResult -Id 'C8' -Description 'Retire authority = 1 and AudioEngine fallback queue symbols = 0' -Status $c8Status -Evidence $c8Evidence)) | Out-Null

# C9: Snapshot authority = 0 (heuristic by descriptor/inventory tokens)
$c9Status = 'fail'
$c9Evidence = 'missing AudioEngine.h'
if (Test-Path -LiteralPath $audioEngineH) {
    $audioText = Get-Content -LiteralPath $audioEngineH -Raw -Encoding UTF8
    $snapshotAuthorityByInventory = ([regex]::Matches($audioText, '\{"[^"\r\n]*snapshot[^"\r\n]*",\s*convo::isr::RuntimeAuthorityClass::Authoritative\}', [System.Text.RegularExpressions.RegexOptions]::IgnoreCase)).Count
    $snapshotAuthorityByDescriptor = ([regex]::Matches($audioText, '\{"[^"\r\n]*snapshot[^"\r\n]*",\s*convo::isr::SemanticCategory::Authority', [System.Text.RegularExpressions.RegexOptions]::IgnoreCase)).Count
    if ($snapshotAuthorityByInventory -eq 0 -and $snapshotAuthorityByDescriptor -eq 0) {
        $c9Status = 'pass'
    }
    $c9Evidence = "snapshotInventoryAuthoritativeCount=$snapshotAuthorityByInventory snapshotDescriptorAuthorityCount=$snapshotAuthorityByDescriptor"
}
$results.Add((New-CheckResult -Id 'C9' -Description 'Snapshot authority = 0' -Status $c9Status -Evidence $c9Evidence)) | Out-Null

# C10: Runtime authority declarations confined to RuntimeState inventory
$c10AuthoritativeMatches = New-Object System.Collections.Generic.List[string]
foreach ($file in $files) {
    $text = Get-Content -LiteralPath $file.FullName -Raw -Encoding UTF8
    if ($text -match 'RuntimeAuthorityClass::Authoritative') {
        $c10AuthoritativeMatches.Add($file.FullName) | Out-Null
    }
}
$c10UniqueFiles = @($c10AuthoritativeMatches | Sort-Object -Unique)
$c10AllowedFilePatterns = @(
    '*src\audioengine\AudioEngine.h',
    '*src\audioengine\RuntimeGraph.h'
)
$c10HasAudioEngineAuthority = ($c10UniqueFiles | Where-Object { $_ -like '*src\audioengine\AudioEngine.h' }).Count -ge 1
$c10OnlyAllowedFiles = $true
foreach ($path in $c10UniqueFiles) {
    $matchedAllowed = $false
    foreach ($pattern in $c10AllowedFilePatterns) {
        if ($path -like $pattern) {
            $matchedAllowed = $true
            break
        }
    }

    if (-not $matchedAllowed) {
        $c10OnlyAllowedFiles = $false
        break
    }
}

$c10Status = if ($c10HasAudioEngineAuthority -and $c10OnlyAllowedFiles) { 'pass' } else { 'fail' }
$c10Evidence = "authoritativeDeclFiles=$($c10UniqueFiles.Count) files=$($c10UniqueFiles -join ';')"
$results.Add((New-CheckResult -Id 'C10' -Description 'Runtime authority = RuntimeState only (inventory confinement)' -Status $c10Status -Evidence $c10Evidence)) | Out-Null

# C11/C12/C13: convolver legacy symbols = 0
$convolverRoot = Join-Path $repoRootResolved 'src\convolver'
$convolverFiles = New-Object System.Collections.Generic.List[System.IO.FileInfo]
if (Test-Path -LiteralPath $convolverRoot) {
    Get-ChildItem -LiteralPath $convolverRoot -Recurse -File -Include *.h, *.hpp, *.cpp, *.cxx, *.cc | ForEach-Object {
        $convolverFiles.Add($_) | Out-Null
    }
}

$convolverHeaderPath = Join-Path $repoRootResolved 'src\ConvolverProcessor.h'
if (Test-Path -LiteralPath $convolverHeaderPath) {
    $convolverFiles.Add((Get-Item -LiteralPath $convolverHeaderPath)) | Out-Null
}

$coreRoot = Join-Path $repoRootResolved 'src\core'
if (Test-Path -LiteralPath $coreRoot) {
    Get-ChildItem -LiteralPath $coreRoot -Recurse -File -Include *.h, *.hpp, *.cpp, *.cxx, *.cc | ForEach-Object {
        $convolverFiles.Add($_) | Out-Null
    }
}

$convolverFiles = @($convolverFiles)
$c11 = Count-RegexMatches -Files $convolverFiles -Pattern '\bSafeStateSwapper\b'
$c12 = Count-RegexMatches -Files $convolverFiles -Pattern '\bPendingParams\b'
$c13 = Count-RegexMatches -Files $convolverFiles -Pattern '\bPreparedIRState\b'
$c11Status = if ($c11 -eq 0) { 'pass' } else { 'fail' }
$c12Status = if ($c12 -eq 0) { 'pass' } else { 'fail' }
$c13Status = if ($c13 -eq 0) { 'pass' } else { 'fail' }
$results.Add((New-CheckResult -Id 'C11' -Description 'SafeStateSwapper = 0 (PRT)' -Status $c11Status -Evidence "count=$c11")) | Out-Null
$results.Add((New-CheckResult -Id 'C12' -Description 'PendingParams = 0 (PRT)' -Status $c12Status -Evidence "count=$c12")) | Out-Null
$results.Add((New-CheckResult -Id 'C13' -Description 'PreparedIRState = 0 (PRT)' -Status $c13Status -Evidence "count=$c13")) | Out-Null

# C15: EQ fallback queue concrete fields removed (fail-closed for alias names)
$eqRoot = Join-Path $repoRootResolved 'src\eqprocessor'
$eqFiles = if (Test-Path -LiteralPath $eqRoot) {
    Get-ChildItem -LiteralPath $eqRoot -Recurse -File -Include *.h, *.hpp, *.cpp, *.cxx, *.cc
}
else {
    @()
}
$c15 = Count-RegexMatches -Files $eqFiles -Pattern '\b(?:deferredDeleteFallbackQueue|deferredRetireFallbackQueue_)\b'
$c15Status = if ($c15 -eq 0) { 'pass' } else { 'fail' }
$results.Add((New-CheckResult -Id 'C15' -Description 'EQ fallback queue concrete symbols = 0 (PRT)' -Status $c15Status -Evidence "count=$c15")) | Out-Null

$passCount = @($results | Where-Object { $_.status -eq 'pass' }).Count
$failCount = @($results | Where-Object { $_.status -eq 'fail' }).Count
$manualCount = @($results | Where-Object { $_.status -eq 'manual' }).Count

$report = New-Object PSObject
$report | Add-Member -NotePropertyName 'schema' -NotePropertyValue 'c1_c15_minimal_report_v1'
$report | Add-Member -NotePropertyName 'generatedAt' -NotePropertyValue ((Get-Date).ToString('o'))
$report | Add-Member -NotePropertyName 'source' -NotePropertyValue '.github/scripts/isr-verify-c1-c15-minimal.ps1'
$report | Add-Member -NotePropertyName 'passCount' -NotePropertyValue $passCount
$report | Add-Member -NotePropertyName 'failCount' -NotePropertyValue $failCount
$report | Add-Member -NotePropertyName 'manualCount' -NotePropertyValue $manualCount
$report | Add-Member -NotePropertyName 'checks' -NotePropertyValue ($results.ToArray())

$report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $reportPath -Encoding UTF8

Write-Host "[INFO] report: $reportPath"
if ($failCount -gt 0) {
    throw "c1-c15 minimal verification failed: failCount=$failCount manualCount=$manualCount"
}

Write-Host "[PASS] c1-c15 minimal verification passed (manual checks pending=$manualCount)"
