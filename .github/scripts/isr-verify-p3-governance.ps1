$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$audioRoot = Join-Path $repoRoot "src\audioengine"

function Read-Text([string]$path) {
    if (-not (Test-Path $path)) {
        throw "Required file not found: $path"
    }
    return Get-Content -Path $path -Raw -Encoding UTF8
}

function Assert-Match([string]$text, [string]$pattern, [string]$message) {
    if (-not [regex]::IsMatch($text, $pattern, [System.Text.RegularExpressions.RegexOptions]::Multiline)) {
        throw $message
    }
}

function Assert-NotMatch([string]$text, [string]$pattern, [string]$message) {
    if ([regex]::IsMatch($text, $pattern, [System.Text.RegularExpressions.RegexOptions]::Multiline)) {
        throw $message
    }
}

# R13: publish graph immutable facade checks
$audioEngineHeaderPath = Join-Path $audioRoot "AudioEngine.h"
$audioEngineHeader = Read-Text $audioEngineHeaderPath
$runtimeStateMatch = [regex]::Match(
    $audioEngineHeader,
    'struct\s+RuntimeState\s*:[\s\S]*?\{([\s\S]*?)\};\s*\r?\n\s*using\s+RuntimePublishWorld',
    [System.Text.RegularExpressions.RegexOptions]::Multiline)

if (-not $runtimeStateMatch.Success) {
    throw "R13 gate: RuntimeState block not found in AudioEngine.h"
}

$runtimeStateBody = $runtimeStateMatch.Groups[1].Value
Assert-NotMatch $runtimeStateBody '\bmutable\b' 'R13 gate: RuntimeState must not contain mutable members'
Assert-NotMatch $runtimeStateBody 'std::mutex|std::shared_ptr|std::atomic\s*<' 'R13 gate: RuntimeState must not contain mutex/shared_ptr/atomic'
Assert-NotMatch $runtimeStateBody 'lazy|Lazy|init\s*\(' 'R13 gate: RuntimeState must not include lazy-init patterns'

# R19 + R23: capability-first coordinator API and 2-world boundary fixed
$coordinatorHeaderPath = Join-Path $audioRoot "ISRRuntimePublicationCoordinator.h"
$coordinatorHeader = Read-Text $coordinatorHeaderPath

Assert-Match $coordinatorHeader 'void\s+commit\s*\(\s*PublishAuthority\s*,\s*RuntimeBoundary\s+boundary\s*,\s*const\s+void\*\s+newWorld\s*,\s*std::uint64_t\s+version\s*\)' 'R19 gate: commit must require PublishAuthority and RuntimeBoundary'
Assert-Match $coordinatorHeader 'void\s+retire\s*\(\s*RetireAuthority\s*,\s*RuntimeBoundary\s+boundary\s*,\s*const\s+void\*\s+oldWorld\s*\)' 'R19 gate: retire must require RetireAuthority and RuntimeBoundary'

Assert-NotMatch $coordinatorHeader 'void\s+commit\s*\(\s*RuntimeBoundary\s+boundary' 'R19 gate: authority-less commit overload is forbidden'
Assert-NotMatch $coordinatorHeader 'void\s+retire\s*\(\s*RuntimeBoundary\s+boundary' 'R19 gate: authority-less retire overload is forbidden'

$boundaryMatch = [regex]::Match(
    $coordinatorHeader,
    'enum\s+class\s+RuntimeBoundary\s*:[^{]+\{([\s\S]*?)\}',
    [System.Text.RegularExpressions.RegexOptions]::Multiline)
if (-not $boundaryMatch.Success) {
    throw 'R23 gate: RuntimeBoundary enum not found'
}

$boundaryBody = $boundaryMatch.Groups[1].Value
Assert-Match $boundaryBody '\bRTWorld\b' 'R23 gate: RuntimeBoundary must include RTWorld'
Assert-Match $boundaryBody '\bNonRTWorld\b' 'R23 gate: RuntimeBoundary must include NonRTWorld'
$boundaryNames = @([regex]::Matches($boundaryBody, '\b([A-Za-z_][A-Za-z0-9_]*)\b') | ForEach-Object { $_.Groups[1].Value } | Where-Object { $_ -notin @('uint8_t') })
$uniqueBoundaryNames = $boundaryNames | Select-Object -Unique
if ($uniqueBoundaryNames.Count -ne 2 -or ($uniqueBoundaryNames -notcontains 'RTWorld') -or ($uniqueBoundaryNames -notcontains 'NonRTWorld')) {
    throw "R23 gate: RuntimeBoundary must be exactly { RTWorld, NonRTWorld }"
}

# Additional R23 federation drift check
$federationMatches = Get-ChildItem -Path $audioRoot -Recurse -File -Include *.h, *.hpp, *.cpp, *.cxx, *.cc |
Select-String -Pattern 'Federation|Federated|FullFederation' -SimpleMatch
if ($federationMatches) {
    throw 'R23 gate: federation runtime related tokens detected'
}

# R20: lifecycle normalization regression checks
$lifecycleCppPath = Join-Path $audioRoot "ISRLifecycle.cpp"
$lifecycleCpp = Read-Text $lifecycleCppPath
Assert-Match $lifecycleCpp 'duplicatePrepareCollapsed_' 'R20 gate: HC-1 duplicate prepare collapse instrumentation missing'
Assert-Match $lifecycleCpp 'currentPhase\s*==\s*LifecyclePhase::Uninitialized\s*\|\|\s*currentPhase\s*==\s*LifecyclePhase::Preparing' 'R20 gate: HC-2 release-before-prepare reject is missing'
Assert-Match $lifecycleCpp 'currentPhase\s*==\s*LifecyclePhase::Releasing' 'R20 gate: HC-3 callback during Releasing reject is missing'

# R21: DSPHandle callback view + crossfade completion checks
$audioBlockCppPath = Join-Path $audioRoot "AudioEngine.Processing.AudioBlock.cpp"
$audioBlockCpp = Read-Text $audioBlockCppPath
Assert-Match $audioBlockCpp 'dspHandleRuntime_\.getActiveDSP\s*\(' 'R21 gate: callback active DSP handle view missing'
Assert-Match $audioBlockCpp 'dspHandleRuntime_\.getFadingDSP\s*\(' 'R21 gate: callback fading DSP handle view missing'

$timerCppPath = Join-Path $audioRoot "AudioEngine.Timer.cpp"
$timerCpp = Read-Text $timerCppPath
Assert-Match $timerCpp 'dspHandleRuntime_\.endCrossfade\s*\(' 'R21 gate: endCrossfade handling missing'
Assert-Match $timerCpp 'crossfadeAuthorityRuntime_\.unregisterCrossfade\s*\(' 'R21 gate: crossfade authority unregister missing'

# R14: retire-intent API naming consistency
$commitCppPath = Join-Path $audioRoot "AudioEngine.Commit.cpp"
$commitCpp = Read-Text $commitCppPath
Assert-Match $commitCpp 'retireRuntime_\.emitRetireIntentRT\s*\(' 'R14 gate: retire intent RT API naming consistency missing'
Assert-NotMatch $commitCpp 'retireRuntime_\.emitRetireIntent\s*\(' 'R14 gate: non-RT retire intent API usage detected in commit path'

Write-Host '[PASS] P3 governance gates (R13/R14/R19/R20/R21/R23)'
