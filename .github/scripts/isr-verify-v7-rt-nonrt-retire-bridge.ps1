$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$complianceScriptPath = Join-Path $repoRoot ".github\scripts\check-list-compliance.ps1"
$retireRuntimeExCppPath = Join-Path $repoRoot "src\audioengine\ISRRetireRuntimeEx.cpp"
$commitCppPath = Join-Path $repoRoot "src\audioengine\AudioEngine.Commit.cpp"

foreach ($path in @($complianceScriptPath, $retireRuntimeExCppPath, $commitCppPath)) {
    if (-not (Test-Path $path)) {
        throw "Missing file: $path"
    }
}

$complianceScriptText = Get-Content -LiteralPath $complianceScriptPath -Raw -Encoding UTF8
$retireRuntimeExCppText = Get-Content -LiteralPath $retireRuntimeExCppPath -Raw -Encoding UTF8
$commitCppText = Get-Content -LiteralPath $commitCppPath -Raw -Encoding UTF8

if ($complianceScriptText -notmatch 'rtRetirePattern\s*=') {
    throw 'list compliance script must define rtRetirePattern rule for enqueueRetire scanning.'
}

if (-not $complianceScriptText.Contains('enqueueRetire\s*\(')) {
    throw 'list compliance script must contain RT direct enqueueRetire detection pattern.'
}

if ($complianceScriptText -notmatch "RuleId\s*'2\.7'" -and $complianceScriptText -notmatch "-RuleId\s*'2\.7'") {
    throw 'list compliance script must enforce Rule 2.7 for RT direct enqueue detection.'
}

if ($retireRuntimeExCppText -notmatch 'void\s+RetireRuntimeEx::enqueueRetire\(std::uint32_t\s+slot\)\s*\{[\s\S]*?ASSERT_NON_RT_THREAD\(\);') {
    throw 'RetireRuntimeEx::enqueueRetire must be guarded by ASSERT_NON_RT_THREAD().'
}

if ($commitCppText -notmatch 'void\s+AudioEngine::onRuntimeRetiredNonRt\(const\s+RuntimePublishWorld\*\s+world\)\s+noexcept\s*\{[\s\S]*?ASSERT_NON_RT_THREAD\(\);') {
    throw 'onRuntimeRetiredNonRt must enforce ASSERT_NON_RT_THREAD().'
}

if ($commitCppText -notmatch 'retireRuntime_\.emitRetireIntentRT\(') {
    throw 'R9 bridge path must emit retire intent from callback detection path.'
}

if ($commitCppText -notmatch 'retireRuntimeEx_\.enqueueRetire\(') {
    throw 'R9 bridge path must route retire enqueue through retireRuntimeEx_.enqueueRetire.'
}

Write-Host '[PASS] R9 RT-detect to NonRT-retire bridge policy verified'
