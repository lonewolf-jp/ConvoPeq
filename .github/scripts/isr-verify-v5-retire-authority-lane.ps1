$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$authorityDocPath = Join-Path $repoRoot "doc\work\ISR_Retire_Authority_Graph.md"
$coordinatorHeaderPath = Join-Path $repoRoot "src\audioengine\ISRRuntimePublicationCoordinator.h"
$coordinatorCppPath = Join-Path $repoRoot "src\audioengine\ISRRuntimePublicationCoordinator.cpp"
$retireLaneHeaderPath = Join-Path $repoRoot "src\audioengine\ISRRetireLane.h"
$retireRuntimeHeaderPath = Join-Path $repoRoot "src\audioengine\ISRRetireRuntimeEx.h"
$retireRuntimeCppPath = Join-Path $repoRoot "src\audioengine\ISRRetireRuntimeEx.cpp"
$audioEngineCommitPath = Join-Path $repoRoot "src\audioengine\AudioEngine.Commit.cpp"

foreach ($path in @($authorityDocPath, $coordinatorHeaderPath, $coordinatorCppPath, $retireLaneHeaderPath, $retireRuntimeHeaderPath, $retireRuntimeCppPath, $audioEngineCommitPath)) {
    if (-not (Test-Path $path)) {
        throw "Missing file: $path"
    }
}

$authorityDocText = Get-Content -LiteralPath $authorityDocPath -Raw -Encoding UTF8
$coordinatorHeaderText = Get-Content -LiteralPath $coordinatorHeaderPath -Raw -Encoding UTF8
$coordinatorCppText = Get-Content -LiteralPath $coordinatorCppPath -Raw -Encoding UTF8
$retireLaneHeaderText = Get-Content -LiteralPath $retireLaneHeaderPath -Raw -Encoding UTF8
$retireRuntimeHeaderText = Get-Content -LiteralPath $retireRuntimeHeaderPath -Raw -Encoding UTF8
$retireRuntimeCppText = Get-Content -LiteralPath $retireRuntimeCppPath -Raw -Encoding UTF8
$audioEngineCommitText = Get-Content -LiteralPath $audioEngineCommitPath -Raw -Encoding UTF8

$requiredDocPhrases = @(
    '1 object family = 1 retire authority',
    'Audio Thread',
    'RuntimeWorldRetireManager',
    'retire queue'
)

foreach ($phrase in $requiredDocPhrases) {
    if ($authorityDocText -notmatch [regex]::Escape($phrase)) {
        throw "Retire authority graph is missing required rule text: $phrase"
    }
}

if ($coordinatorHeaderText -notmatch 'enum class RetireAuthority\s*:\s*uint8_t\s*\{\s*Granted\s*=\s*1\s*\};') {
    throw 'RetireAuthority must be defined as single canonical capability (Granted=1).'
}

if ($coordinatorHeaderText -notmatch 'void retire\(RetireAuthority,\s*RuntimeBoundary\s*boundary,\s*const void\*\s*oldWorld\);') {
    throw 'RuntimePublicationCoordinator::retire signature must require RetireAuthority capability.'
}

if ($coordinatorCppText -notmatch 'void RuntimePublicationCoordinator::retire\(RetireAuthority,') {
    throw 'RuntimePublicationCoordinator::retire implementation missing RetireAuthority-typed entrypoint.'
}

if ($audioEngineCommitText -notmatch 'runtimePublicationCoordinator_\.retire\(convo::isr::RetireAuthority::Granted,' `
    -and $audioEngineCommitText -notmatch 'runtimePublicationBridge_\.retire\(convo::isr::RetireAuthority::Granted,' `
    -and $audioEngineCommitText -notmatch '\bretireRuntimePublication\s*\(\s*world\s*\)') {
    throw 'AudioEngine non-RT retire path must call retire authority lane with RetireAuthority::Granted (directly or via wrapper/bridge).'
}

$allSource = Get-ChildItem -Path (Join-Path $repoRoot 'src') -Recurse -File -Include *.h,*.hpp,*.cpp,*.cxx,*.cc |
    ForEach-Object { Get-Content -LiteralPath $_.FullName -Raw -Encoding UTF8 }

$retireAuthorityDeclCount = ([regex]::Matches(($allSource -join "`n"), 'enum\s+class\s+RetireAuthority\b')).Count
if ($retireAuthorityDeclCount -ne 1) {
    throw "RetireAuthority declaration count must be exactly 1. actual=$retireAuthorityDeclCount"
}

$requiredLanes = @('RTIntent', 'Coordination', 'Epoch', 'Reclaim', 'Quarantine')
foreach ($lane in $requiredLanes) {
    if ($retireLaneHeaderText -notmatch ('\b' + [regex]::Escape($lane) + '\b')) {
        throw "RetireLane is missing required lane: $lane"
    }
}

$requiredLaneOps = @('emitIntent\(', 'enqueueRetire\(', 'settleEpoch\(', 'reclaim\(', 'quarantine\(', 'laneOf\(')
foreach ($token in $requiredLaneOps) {
    if ($retireRuntimeHeaderText -notmatch $token) {
        throw "RetireRuntimeEx header is missing required lane operation: $token"
    }
}

if ($retireRuntimeCppText -notmatch 'ASSERT_NON_RT_THREAD\(\);') {
    throw 'RetireRuntimeEx must enforce Non-RT guard on retire enqueue path.'
}

Write-Host '[PASS] R4 retire authority identity and lane split policy verified'
