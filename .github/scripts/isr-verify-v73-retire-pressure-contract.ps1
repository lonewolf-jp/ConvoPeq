Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$evidenceDir = Join-Path $repoRoot 'evidence'
$reportPath = Join-Path $evidenceDir 'isr_v73_retire_pressure_report.json'

if (-not (Test-Path -LiteralPath $evidenceDir)) {
    New-Item -ItemType Directory -Path $evidenceDir -Force | Out-Null
}

$targets = [ordered]@{
    authority = Join-Path $repoRoot 'src\audioengine\ISRAuthorityClass.h'
    header    = Join-Path $repoRoot 'src\audioengine\AudioEngine.h'
    threading = Join-Path $repoRoot 'src\audioengine\AudioEngine.Retire.cpp'
    release   = Join-Path $repoRoot 'src\audioengine\AudioEngine.Processing.ReleaseResources.cpp'
}

foreach ($kv in $targets.GetEnumerator()) {
    if (-not (Test-Path -LiteralPath $kv.Value)) {
        throw "Missing required source file: $($kv.Value)"
    }
}

$authorityText = Get-Content -LiteralPath $targets.authority -Raw -Encoding UTF8
$headerText = Get-Content -LiteralPath $targets.header -Raw -Encoding UTF8
$threadingText = Get-Content -LiteralPath $targets.threading -Raw -Encoding UTF8
$releaseText = Get-Content -LiteralPath $targets.release -Raw -Encoding UTF8

$violations = New-Object 'System.Collections.Generic.List[object]'

function Add-Violation {
    param(
        [Parameter(Mandatory = $true)][string]$CheckId,
        [Parameter(Mandatory = $true)][string]$File,
        [Parameter(Mandatory = $true)][string]$Message
    )

    $violations.Add(@{
            checkId = $CheckId
            file    = $File
            message = $Message
        }) | Out-Null
}

function Assert-Pattern {
    param(
        [Parameter(Mandatory = $true)][string]$Text,
        [Parameter(Mandatory = $true)][string]$Pattern,
        [Parameter(Mandatory = $true)][string]$CheckId,
        [Parameter(Mandatory = $true)][string]$File,
        [Parameter(Mandatory = $true)][string]$Message
    )

    if (-not [System.Text.RegularExpressions.Regex]::IsMatch($Text, $Pattern, [System.Text.RegularExpressions.RegexOptions]::Singleline)) {
        Add-Violation -CheckId $CheckId -File $File -Message $Message
    }
}

# CI-RETIREPRESS-001: enqueue失敗時にQueuePressure/QueueFullを返し silent drop しない
#   [P0-5] single-attempt + drop pattern: m_retireRouter->enqueueRetire -> success or QueuePressure
#   Legacy two-attempt pattern (enqueueRetireEpochBounded x2) も許容する。
#   ★ Practical Stable: enqueueWithRetry (Bug2 Phase1 Router 集約) も許容する。
$hasTwoAttemptPattern = [System.Text.RegularExpressions.Regex]::IsMatch($headerText,
    'enqueueRetireEpochBounded\(ptr,\s*deleter,\s*epoch\).*enqueueRetireEpochBounded\(ptr,\s*deleter,\s*epoch\)',
    [System.Text.RegularExpressions.RegexOptions]::Singleline)
$hasSingleAttemptPressureReturn = [System.Text.RegularExpressions.Regex]::IsMatch($headerText,
    'RetireEnqueueResult::(QueuePressure|QueueFull|Shutdown)',
    [System.Text.RegularExpressions.RegexOptions]::Singleline)
$hasEnqueueWithRetry = [System.Text.RegularExpressions.Regex]::IsMatch($headerText,
    'enqueueWithRetry\(ptr,\s*deleter,\s*epoch,',
    [System.Text.RegularExpressions.RegexOptions]::Singleline)
$hasLegacyFallbackEnqueue = [System.Text.RegularExpressions.Regex]::IsMatch($headerText, 'deferredDeleteFallbackQueue\.push_back\(DeferredDeleteFallbackEntry\{', [System.Text.RegularExpressions.RegexOptions]::Singleline)

if (-not $hasTwoAttemptPattern -and -not $hasSingleAttemptPressureReturn -and -not $hasEnqueueWithRetry -and -not $hasLegacyFallbackEnqueue) {
    Add-Violation -CheckId 'CI-RETIREPRESS-001' -File 'src/audioengine/AudioEngine.h' -Message 'enqueueDeferredDeleteNonRt must either (a) two-attempt enqueueRetireEpochBounded, (b) single-attempt with QueuePressure return, (c) enqueueWithRetry (Router集約), or (d) legacy fallback enqueue.'
}

# CI-RETIREPRESS-001(b): reclaimRetired is only required for two-attempt pattern
if ($hasTwoAttemptPattern) {
    Assert-Pattern -Text $headerText -Pattern 'm_epochDomain\.reclaimRetired\(\);' -CheckId 'CI-RETIREPRESS-001' -File 'src/audioengine/AudioEngine.h' -Message 'enqueueDeferredDeleteNonRt must call reclaimRetired() before second enqueue attempt.'
}

# CI-RETIREPRESS-002: fallback蓄積時はbest-effort drainキックが必須
Assert-Pattern -Text $headerText -Pattern 'drainDeferredRetireQueues\(false\);' -CheckId 'CI-RETIREPRESS-002' -File 'src/audioengine/AudioEngine.h' -Message 'enqueueDeferredDeleteNonRt must trigger best-effort deferred drain after fallback enqueue.'

# CI-RETIREPRESS-003: shutdown中の通常経路drain抑止（allowDuringShutdownでのみ許可）
Assert-Pattern -Text $threadingText -Pattern 'if\s*\(!allowDuringShutdown\s*&&\s*isShutdownInProgress\(\)\)\s*\n\s*return;' -CheckId 'CI-RETIREPRESS-003' -File 'src/audioengine/AudioEngine.Threading.cpp' -Message 'drainDeferredRetireQueues must gate non-shutdown drains while shutdown is in progress.'

# CI-RETIREPRESS-004: pressure telemetry/backlog publish が存在する
Assert-Pattern -Text $threadingText -Pattern 'setRetireBacklogCount\(retireDepth\)' -CheckId 'CI-RETIREPRESS-004' -File 'src/audioengine/AudioEngine.Threading.cpp' -Message 'Retire backlog count telemetry publication is missing.'
Assert-Pattern -Text $threadingText -Pattern 'setDeferredRetireResidencyCount\(fallbackDepth\)' -CheckId 'CI-RETIREPRESS-004' -File 'src/audioengine/AudioEngine.Threading.cpp' -Message 'Deferred retire residency telemetry publication is missing.'

# CI-RETIREPRESS-005: shutdown/release pathで強制drainが実行される
Assert-Pattern -Text $releaseText -Pattern 'drainDeferredRetireQueues\(true\);' -CheckId 'CI-RETIREPRESS-005' -File 'src/audioengine/AudioEngine.Processing.ReleaseResources.cpp' -Message 'releaseResources must force-drain deferred retire queues during shutdown.'
Assert-Pattern -Text $releaseText -Pattern 'pendingRetireCount' -CheckId 'CI-RETIREPRESS-005' -File 'src/audioengine/AudioEngine.Processing.ReleaseResources.cpp' -Message 'releaseResources must account pending retire count for bounded teardown diagnostics.'

# CI-RETIREPRESS-006: RetireEnqueueResult 4分岐を要求
Assert-Pattern -Text $authorityText -Pattern 'enum class RetireEnqueueResult\s*:\s*std::uint8_t\s*\{\s*Success\s*=\s*0\s*,\s*QueuePressure\s*,\s*QueueFull\s*,\s*Shutdown\s*\};' -CheckId 'CI-RETIREPRESS-006' -File 'src/audioengine/ISRAuthorityClass.h' -Message 'RetireEnqueueResult must define Success/QueuePressure/QueueFull/Shutdown.'
Assert-Pattern -Text $headerText -Pattern 'enqueueDeferredDeleteNonRtWithResult\(void\* ptr,\s*void \(\*deleter\)\(void\*\)\)\s*noexcept' -CheckId 'CI-RETIREPRESS-006' -File 'src/audioengine/AudioEngine.h' -Message 'AudioEngine must provide enqueueDeferredDeleteNonRtWithResult helper.'

$report = @{
    schema         = 'isr_v73_retire_pressure_report_v1'
    generatedAt    = (Get-Date -Format 'o')
    checks         = @(
        'CI-RETIREPRESS-001',
        'CI-RETIREPRESS-002',
        'CI-RETIREPRESS-003',
        'CI-RETIREPRESS-004',
        'CI-RETIREPRESS-005',
        'CI-RETIREPRESS-006'
    )
    violationCount = $violations.Count
    violations     = $violations.ToArray()
}

$report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $reportPath -Encoding UTF8
Write-Host "[INFO] wrote $reportPath"

if ($violations.Count -gt 0) {
    foreach ($violation in $violations) {
        Write-Host "[ERROR] [$($violation.checkId)] $($violation.file) $($violation.message)"
    }

    throw "ISR v7.3 retire pressure checks failed. violations=$($violations.Count)"
}

Write-Host '[PASS] ISR v7.3 retire pressure checks passed'
