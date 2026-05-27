$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$hbSpecPath = Join-Path $repoRoot "doc\work\ISR_HB_Graph_Specification.md"
$lifecycleHeaderPath = Join-Path $repoRoot "src\audioengine\ISRLifecycle.h"
$lifecycleCppPath = Join-Path $repoRoot "src\audioengine\ISRLifecycle.cpp"
$prepareCppPath = Join-Path $repoRoot "src\audioengine\AudioEngine.Processing.PrepareToPlay.cpp"
$audioBlockCppPath = Join-Path $repoRoot "src\audioengine\AudioEngine.Processing.AudioBlock.cpp"
$releaseCppPath = Join-Path $repoRoot "src\audioengine\AudioEngine.Processing.ReleaseResources.cpp"

foreach ($path in @($hbSpecPath, $lifecycleHeaderPath, $lifecycleCppPath, $prepareCppPath, $audioBlockCppPath, $releaseCppPath)) {
    if (-not (Test-Path $path)) {
        throw "Missing file: $path"
    }
}

$hbSpecText = Get-Content -LiteralPath $hbSpecPath -Raw -Encoding UTF8
$lifecycleHeaderText = Get-Content -LiteralPath $lifecycleHeaderPath -Raw -Encoding UTF8
$lifecycleCppText = Get-Content -LiteralPath $lifecycleCppPath -Raw -Encoding UTF8
$prepareCppText = Get-Content -LiteralPath $prepareCppPath -Raw -Encoding UTF8
$audioBlockCppText = Get-Content -LiteralPath $audioBlockCppPath -Raw -Encoding UTF8
$releaseCppText = Get-Content -LiteralPath $releaseCppPath -Raw -Encoding UTF8

$requiredDomainFPatterns = @(
    'Domain\s+F:\s+Parameter\s+Smoothing\s*/\s*Audio\s+Callback\s+Sync',
    'F2:\s*`?prepareToPlay\(\)`?\s*complete\s*->\s*first\s+audio\s+callback\s+start',
    '`?prepareToPlay\(\)`?\s*未完了状態で\s*audio\s+callback\s+が\s*publish\s+payload\s*を\s*参照してはならない',
    'host\s+callback\s+由来の再入可能経路は\s*Domain\s+F\s*内で\s*単一順序規約に従う'
)

foreach ($pattern in $requiredDomainFPatterns) {
    if (-not [regex]::IsMatch($hbSpecText, $pattern, [System.Text.RegularExpressions.RegexOptions]::Multiline)) {
        throw "HB Domain F specification missing required semantic pattern: $pattern"
    }
}

$requiredLifecycleHeaderTokens = @(
    'LifecyclePhase',
    'Preparing',
    'Prepared',
    'AudioRunning',
    'Releasing',
    'enterPrepare\(',
    'enterAudioCallback\(',
    'enterRelease\('
)

foreach ($token in $requiredLifecycleHeaderTokens) {
    if ($lifecycleHeaderText -notmatch $token) {
        throw "Lifecycle header missing token required for Domain F enforcement: $token"
    }
}

if ($prepareCppText -notmatch 'ASSERT_NON_RT_THREAD\(\);') {
    throw 'prepareToPlay must enforce Non-RT thread guard.'
}
if ($prepareCppText -notmatch 'lifecycleRuntime_\.enterPrepare\(' -or $prepareCppText -notmatch 'lifecycleRuntime_\.leavePrepare\(') {
    throw 'prepareToPlay must bracket execution with lifecycle enterPrepare/leavePrepare.'
}

if ($audioBlockCppText -notmatch 'lifecycleRuntime_\.enterAudioCallback\(' -or $audioBlockCppText -notmatch 'lifecycleRuntime_\.leaveAudioCallback\(') {
    throw 'audio callback must bracket execution with lifecycle enterAudioCallback/leaveAudioCallback.'
}

if ($releaseCppText -notmatch 'ASSERT_NON_RT_THREAD\(\);') {
    throw 'releaseResources must enforce Non-RT thread guard.'
}
if ($releaseCppText -notmatch 'lifecycleRuntime_\.enterRelease\(' -or $releaseCppText -notmatch 'lifecycleRuntime_\.leaveRelease\(') {
    throw 'releaseResources must bracket execution with lifecycle enterRelease/leaveRelease.'
}

if ($lifecycleCppText -notmatch 'currentPhase\s*==\s*LifecyclePhase::Releasing\s*\|\|' -or $lifecycleCppText -notmatch 'currentPhase\s*==\s*LifecyclePhase::Released') {
    throw 'Lifecycle runtime must reject callback entry during release/released phase.'
}

if ($lifecycleCppText -notmatch 'currentPhase\s*==\s*LifecyclePhase::Uninitialized\s*\|\|\s*currentPhase\s*==\s*LifecyclePhase::Preparing') {
    throw 'Lifecycle runtime must reject release entry before prepare completion.'
}

if ($lifecycleCppText -notmatch 'case\s+LifecyclePhase::AudioRunning:[\s\S]*?valid\s*=\s*\(to\s*==\s*LifecyclePhase::Prepared\s*\|\|\s*to\s*==\s*LifecyclePhase::Shutdown\);') {
    throw 'Lifecycle transition table must keep AudioRunning single-order transition (Prepared or Shutdown only).'
}

Write-Host '[PASS] R5 Domain F callback ordering and reentrancy policy verified'
