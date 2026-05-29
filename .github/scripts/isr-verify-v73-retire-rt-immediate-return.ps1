Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$evidenceDir = Join-Path $repoRoot 'evidence'
$reportPath = Join-Path $evidenceDir 'isr_v73_retire_rt_immediate_return_report.json'

if (-not (Test-Path -LiteralPath $evidenceDir)) {
    New-Item -ItemType Directory -Path $evidenceDir -Force | Out-Null
}

$targets = [ordered]@{
    audioBlock = Join-Path $repoRoot 'src\audioengine\AudioEngine.Processing.AudioBlock.cpp'
    blockDouble = Join-Path $repoRoot 'src\audioengine\AudioEngine.Processing.BlockDouble.cpp'
}

foreach ($kv in $targets.GetEnumerator()) {
    if (-not (Test-Path -LiteralPath $kv.Value)) {
        throw "Missing required source file: $($kv.Value)"
    }
}

$audioBlockText = Get-Content -LiteralPath $targets.audioBlock -Raw -Encoding UTF8
$blockDoubleLines = [string[]]@(Get-Content -LiteralPath $targets.blockDouble -Encoding UTF8)
$audioBlockLines = [string[]]@(Get-Content -LiteralPath $targets.audioBlock -Encoding UTF8)

$violations = New-Object 'System.Collections.Generic.List[object]'

function Add-Violation {
    param(
        [Parameter(Mandatory = $true)][string]$CheckId,
        [Parameter(Mandatory = $true)][string]$File,
        [Parameter(Mandatory = $true)][string]$Message,
        [int]$Line = 0,
        [string]$Snippet = ''
    )

    $violations.Add(@{
            checkId = $CheckId
            file = $File
            line = $Line
            message = $Message
            snippet = $Snippet
        }) | Out-Null
}

# CI-RETIRE-RT-001: shutdown 時は RT で即時 return（new work/no block）
$shutdownFastReturnPattern = 'if \(isShutdownInProgress\(\)\)\s*\{\s*shutdownRuntime_\.markLateCallback\(\);\s*bufferToFill\.clearActiveBufferRegion\(\);\s*return;\s*\}'
if (-not [System.Text.RegularExpressions.Regex]::IsMatch($audioBlockText, $shutdownFastReturnPattern, [System.Text.RegularExpressions.RegexOptions]::Singleline)) {
    Add-Violation -CheckId 'CI-RETIRE-RT-001' -File 'src/audioengine/AudioEngine.Processing.AudioBlock.cpp' -Message 'Audio RT path must fast-return under shutdown with clearActiveBufferRegion().'
}

$forbiddenRetireOps = @(
    'retireDSP(',
    'enqueueDeferredDeleteNonRt(',
    'enqueueDeferredDeleteNonRtWithResult(',
    'enqueueRetireEpochBounded(',
    'deferredDeleteFallbackQueue'
)

$forbiddenRtSideEffects = @(
    'juce::Logger::writeToLog(',
    'std::lock_guard<std::mutex>',
    'juce::Thread::sleep('
)

function Test-ForbiddenTokens {
    param(
        [Parameter(Mandatory = $true)][AllowEmptyCollection()][object[]]$Lines,
        [Parameter(Mandatory = $true)][string]$RelativePath,
        [Parameter(Mandatory = $true)][string[]]$Tokens,
        [Parameter(Mandatory = $true)][string]$CheckId,
        [Parameter(Mandatory = $true)][string]$Message
    )

    for ($i = 0; $i -lt $Lines.Count; $i++) {
        $line = [string]$Lines[$i]
        $trim = $line.TrimStart()
        if ($trim.StartsWith('//')) {
            continue
        }

        foreach ($token in $Tokens) {
            if ($line.Contains($token)) {
                Add-Violation -CheckId $CheckId -File $RelativePath -Line ($i + 1) -Message $Message -Snippet $line.Trim()
            }
        }
    }
}

# CI-RETIRE-RT-002: RT pathで retire enqueue 操作を行わない（Non-RT lane へ委譲）
Test-ForbiddenTokens -Lines $audioBlockLines -RelativePath 'src/audioengine/AudioEngine.Processing.AudioBlock.cpp' -Tokens $forbiddenRetireOps -CheckId 'CI-RETIRE-RT-002' -Message 'RT path must not perform retire enqueue/deferred-delete operations.'
Test-ForbiddenTokens -Lines $blockDoubleLines -RelativePath 'src/audioengine/AudioEngine.Processing.BlockDouble.cpp' -Tokens $forbiddenRetireOps -CheckId 'CI-RETIRE-RT-002' -Message 'RT double-path must not perform retire enqueue/deferred-delete operations.'

# CI-RETIRE-RT-003: QueueFull/Shutdown系で RT path に block/alloc/log を混入させない
Test-ForbiddenTokens -Lines $audioBlockLines -RelativePath 'src/audioengine/AudioEngine.Processing.AudioBlock.cpp' -Tokens $forbiddenRtSideEffects -CheckId 'CI-RETIRE-RT-003' -Message 'RT path must not add blocking/lock/log side effects for retire failure handling.'
Test-ForbiddenTokens -Lines $blockDoubleLines -RelativePath 'src/audioengine/AudioEngine.Processing.BlockDouble.cpp' -Tokens $forbiddenRtSideEffects -CheckId 'CI-RETIRE-RT-003' -Message 'RT double-path must not add blocking/lock/log side effects for retire failure handling.'

$report = @{
    schema = 'isr_v73_retire_rt_immediate_return_report_v1'
    generatedAt = (Get-Date -Format 'o')
    checks = @('CI-RETIRE-RT-001', 'CI-RETIRE-RT-002', 'CI-RETIRE-RT-003')
    violationCount = $violations.Count
    violations = $violations.ToArray()
}

$report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $reportPath -Encoding UTF8
Write-Host "[INFO] wrote $reportPath"

if ($violations.Count -gt 0) {
    foreach ($v in $violations) {
        $location = if ($v.line -gt 0) { "$($v.file):$($v.line)" } else { "$($v.file)" }
        Write-Host "[ERROR] [$($v.checkId)] $location $($v.message)"
    }
    throw "ISR v7.3 RT immediate-return checks failed. violations=$($violations.Count)"
}

Write-Host '[PASS] ISR v7.3 RT immediate-return checks passed'
