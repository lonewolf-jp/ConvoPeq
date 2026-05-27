param(
    [string]$LogPath = "build/ConvoPeq_artefacts/Release/ConvoPeq.log",
    [string]$SnapshotPath = ".github/tmp/isr-8_1-metrics-snapshot.json",
    [string]$ReportPath = "evidence/rebuild_admission_8_1_metrics_report.json",
    [switch]$TryAutoCaptureOnMissingLog,
    [int]$AutoCaptureTimeoutSec = 20,
    [switch]$UseDeltaFromSnapshot,
    [switch]$WriteSnapshot,
    [switch]$FailIfNotReady
)

$ErrorActionPreference = 'Stop'

function Test-HasRebuildSignals {
    param([string]$Path)

    if ([string]::IsNullOrWhiteSpace($Path) -or -not (Test-Path -LiteralPath $Path)) {
        return $false
    }

    $signalPattern = 'reason=requestRebuild_sr_bs|reason=task_queued|reason=pending_duplicate|reason=same_as_pending_would_merge|reason=deferred_finalize_ready|reason=deferred_finalize_rebuild_requested|event=REBUILD_FORCED_DISPATCH|policy=MustExecute'
    return [bool](Select-String -Path $Path -Pattern $signalPattern -Quiet)
}

function Resolve-EffectiveLogPath {
    param([string]$RequestedLogPath)

    if (-not [string]::IsNullOrWhiteSpace($RequestedLogPath) -and (Test-Path -LiteralPath $RequestedLogPath)) {
        return [System.IO.Path]::GetFullPath($RequestedLogPath)
    }

    $candidatePaths = @(
        'build/ConvoPeq_artefacts/Release/ConvoPeq.log',
        'build/ConvoPeq_artefacts/Debug/ConvoPeq.log',
        'build/ConvoPeq_artefacts/RelWithDebInfo/ConvoPeq.log',
        'build/ConvoPeq_artefacts/MinSizeRel/ConvoPeq.log'
    )

    $resolvedCandidates = New-Object System.Collections.Generic.List[object]

    foreach ($candidate in $candidatePaths) {
        if (Test-Path -LiteralPath $candidate) {
            $item = Get-Item -LiteralPath $candidate
            $resolvedCandidates.Add([PSCustomObject]@{
                    Path          = $item.FullName
                    LastWriteTime = $item.LastWriteTime
                    HasSignals    = (Test-HasRebuildSignals -Path $item.FullName)
                })
        }
    }

    $buildRoot = 'build'
    if (Test-Path -LiteralPath $buildRoot) {
        $discovered = Get-ChildItem -LiteralPath $buildRoot -Recurse -File -ErrorAction SilentlyContinue |
        Where-Object {
            $_.Name -like 'ConvoPeq*.log' -or
            ($_.Name -like '*.log' -and $_.FullName -match 'ConvoPeq_artefacts')
        }

        foreach ($file in $discovered) {
            if (-not ($resolvedCandidates | Where-Object { $_.Path -eq $file.FullName })) {
                $resolvedCandidates.Add([PSCustomObject]@{
                        Path          = $file.FullName
                        LastWriteTime = $file.LastWriteTime
                        HasSignals    = (Test-HasRebuildSignals -Path $file.FullName)
                    })
            }
        }
    }

    $latestWithSignals = $resolvedCandidates |
    Where-Object { $_.HasSignals } |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

    if ($null -ne $latestWithSignals) {
        return $latestWithSignals.Path
    }

    $latestAny = $resolvedCandidates |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

    if ($null -ne $latestAny) {
        return $latestAny.Path
    }

    return $RequestedLogPath
}

function Get-GenerationGuidance {
    param(
        [string]$ResolvedLogPath,
        [string]$Status,
        [string]$Reason
    )

    $repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
    $tasksPath = Join-Path $repoRoot '.vscode\tasks.json'
    $artefactRoot = Join-Path $repoRoot 'build\ConvoPeq_artefacts'

    $taskHints = @()
    if (Test-Path -LiteralPath $tasksPath) {
        try {
            $tasksJson = Get-Content -LiteralPath $tasksPath -Raw -Encoding UTF8 | ConvertFrom-Json
            foreach ($task in @($tasksJson.tasks)) {
                $label = [string]$task.label
                if ([string]::IsNullOrWhiteSpace($label)) {
                    continue
                }

                if ($label -match 'Release|Debug|Build') {
                    $taskHints += [ordered]@{
                        label = $label
                        group = [string]$task.group
                    }
                }
            }
        }
        catch {
            # guidance is best-effort; do not fail metrics script due to malformed tasks.json
        }
    }

    $exeHints = @()
    $logHints = @()
    $captureCommands = @()
    if (Test-Path -LiteralPath $artefactRoot) {
        $exeCandidates = Get-ChildItem -LiteralPath $artefactRoot -Recurse -File -Filter 'ConvoPeq.exe' -ErrorAction SilentlyContinue |
            Sort-Object LastWriteTime -Descending

        foreach ($exe in $exeCandidates) {
            $exeHints += [ordered]@{
                path = $exe.FullName
                lastWrite = $exe.LastWriteTime.ToString('yyyy-MM-dd HH:mm:ss')
            }

            $logCandidate = Join-Path $exe.DirectoryName 'ConvoPeq.log'
            $logHints += [ordered]@{
                path = $logCandidate
                exists = (Test-Path -LiteralPath $logCandidate)
            }

            $captureCommands += [ordered]@{
                mode = 'launch'
                executablePath = $exe.FullName
                command = "& '$($exe.FullName)'"
                expectedLog = $logCandidate
            }
        }
    }

    $suggestedAction = ''
    if ($Status -eq 'skipped' -and $Reason -eq 'log_not_found') {
        $suggestedAction = 'Run one of the detected Release/Debug build tasks, launch ConvoPeq.exe once to emit logs, then rerun isr-rebuild-admission-8_1-metrics.ps1.'
    }
    elseif ($Status -eq 'evaluated') {
        $suggestedAction = 'If readyToClose8_1 is false, collect a broader runtime window and rerun with -UseDeltaFromSnapshot and -WriteSnapshot.'
    }

    return [ordered]@{
        resolvedLogPath = $ResolvedLogPath
        taskHints = $taskHints
        executableHints = $exeHints
        logHints = $logHints
        captureCommands = $captureCommands
        suggestedAction = $suggestedAction
    }
}

function Invoke-AutoLogCapture {
    param(
        [string]$TargetLogPath,
        [int]$TimeoutSec,
        [hashtable]$Guidance
    )

    $result = [ordered]@{
        attempted = $false
        success = $false
        reason = 'not_requested'
        attemptedCommand = $null
        producedLogPath = $TargetLogPath
        timeoutSec = $TimeoutSec
    }

    if (-not $TryAutoCaptureOnMissingLog) {
        return $result
    }

    $result.attempted = $true
    $result.reason = 'no_capture_command'

    $captureCommands = @()
    if ($null -ne $Guidance -and $null -ne $Guidance.captureCommands) {
        $captureCommands = @($Guidance.captureCommands)
    }

    foreach ($capture in $captureCommands) {
        $command = [string]$capture.command
        $expectedLog = [string]$capture.expectedLog
        $exePath = [string]$capture.executablePath
        if ([string]::IsNullOrWhiteSpace($command) -or [string]::IsNullOrWhiteSpace($expectedLog)) {
            continue
        }

        if ([string]::IsNullOrWhiteSpace($exePath)) {
            if ($command -match "^&\s*'(.+)'$") {
                $exePath = $Matches[1]
            }
        }

        if ([string]::IsNullOrWhiteSpace($exePath) -or -not (Test-Path -LiteralPath $exePath)) {
            continue
        }

        $result.attemptedCommand = $command
        $result.producedLogPath = $expectedLog

        $proc = $null
        try {
            $workDir = Split-Path -Parent $exePath
            $proc = Start-Process -FilePath $exePath -WorkingDirectory $workDir -PassThru
        }
        catch {
            $result.reason = "launch_failed: $($_.Exception.Message)"
            continue
        }

        try {
            $deadline = (Get-Date).AddSeconds([Math]::Max(1, $TimeoutSec))
            while ((Get-Date) -lt $deadline) {
                if (Test-Path -LiteralPath $expectedLog) {
                    $result.success = $true
                    $result.reason = 'captured'
                    break
                }

                Start-Sleep -Milliseconds 500
            }

            if (-not $result.success) {
                $result.reason = 'timeout_waiting_for_log'
            }
        }
        finally {
            if ($null -ne $proc) {
                try {
                    if (-not $proc.HasExited) {
                        Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
                    }
                }
                catch {
                    # best effort cleanup
                }
            }
        }

        if ($result.success) {
            break
        }
    }

    return $result
}

function Write-MetricsReport {
    param(
        [string]$Path,
        [string]$Status,
        [string]$Reason,
        [string]$ResolvedLog,
        [bool]$ReadyToClose,
        [hashtable]$Metrics,
        [hashtable]$Evaluation,
        [hashtable]$Guidance,
        [hashtable]$AutoCapture
    )

    $dir = Split-Path -Parent $Path
    if (-not [string]::IsNullOrWhiteSpace($dir) -and -not (Test-Path -LiteralPath $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }

    $payload = [ordered]@{
        schema          = 'rebuild_admission_8_1_metrics_report_v1'
        generatedAt     = (Get-Date -Format 'o')
        status          = $Status
        reason          = $Reason
        log             = $ResolvedLog
        readyToClose8_1 = $ReadyToClose
        metrics         = $Metrics
        evaluation      = $Evaluation
        guidance        = $Guidance
        autoCapture     = $AutoCapture
    }

    Set-Content -LiteralPath $Path -Value ($payload | ConvertTo-Json -Depth 6) -Encoding UTF8
    Write-Output "report=$Path"
}

$effectiveLogPath = Resolve-EffectiveLogPath -RequestedLogPath $LogPath
$autoCaptureResult = [ordered]@{
    attempted = $TryAutoCaptureOnMissingLog.IsPresent
    success = $false
    reason = 'not_needed'
    attemptedCommand = $null
    producedLogPath = $LogPath
    timeoutSec = $AutoCaptureTimeoutSec
}

if (-not [string]::IsNullOrWhiteSpace($effectiveLogPath)) {
    $LogPath = $effectiveLogPath
    $autoCaptureResult.producedLogPath = $LogPath
}

if (-not (Test-Path -LiteralPath $LogPath)) {
    $guidance = Get-GenerationGuidance -ResolvedLogPath $LogPath -Status 'skipped' -Reason 'log_not_found'
    $autoCapture = Invoke-AutoLogCapture -TargetLogPath $LogPath -TimeoutSec $AutoCaptureTimeoutSec -Guidance $guidance
    $autoCaptureResult = $autoCapture

    if ($autoCapture.success -and (Test-Path -LiteralPath $autoCapture.producedLogPath)) {
        $LogPath = $autoCapture.producedLogPath
    }
    else {
        Write-Output "log=$LogPath"
        Write-Output 'status=skipped'
        Write-Output 'reason=log_not_found'
        Write-Output 'readyToClose8_1=false'
        Write-MetricsReport -Path $ReportPath -Status 'skipped' -Reason 'log_not_found' -ResolvedLog $LogPath -ReadyToClose $false -Metrics ([ordered]@{}) -Evaluation ([ordered]@{}) -Guidance $guidance -AutoCapture $autoCapture
        return
    }
}

$logItem = Get-Item -LiteralPath $LogPath

function Measure-MatchCount {
    param(
        [Parameter(Mandatory = $true)][string]$Pattern,
        [switch]$Regex
    )

    if ($Regex) {
        return @(Select-String -Path $LogPath -Pattern $Pattern).Count
    }

    return @(Select-String -Path $LogPath -Pattern $Pattern -SimpleMatch).Count
}

function Get-Snapshot {
    param([string]$Path)

    if (-not (Test-Path -LiteralPath $Path)) {
        return $null
    }

    $raw = Get-Content -LiteralPath $Path -Raw
    if ([string]::IsNullOrWhiteSpace($raw)) {
        return $null
    }

    return ($raw | ConvertFrom-Json)
}

function Set-Snapshot {
    param(
        [string]$Path,
        [hashtable]$Metrics,
        [string]$SourceLog,
        [string]$LastWrite
    )

    $dir = Split-Path -Parent $Path
    if (-not [string]::IsNullOrWhiteSpace($dir) -and -not (Test-Path -LiteralPath $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }

    $obj = [ordered]@{
        generatedAt  = (Get-Date).ToString('yyyy-MM-dd HH:mm:ss')
        sourceLog    = $SourceLog
        logLastWrite = $LastWrite
        metrics      = $Metrics
    }

    ($obj | ConvertTo-Json -Depth 5) | Set-Content -LiteralPath $Path -Encoding UTF8
}

$metrics = [ordered]@{
    requestRebuild_sr_bs                = Measure-MatchCount -Pattern 'reason=requestRebuild_sr_bs'
    task_queued                         = Measure-MatchCount -Pattern 'reason=task_queued'
    pending_duplicate                   = Measure-MatchCount -Pattern 'reason=pending_duplicate'
    same_as_pending_would_merge         = Measure-MatchCount -Pattern 'reason=same_as_pending_would_merge'
    deferred_finalize_ready             = Measure-MatchCount -Pattern 'reason=deferred_finalize_ready'
    deferred_finalize_rebuild_req       = Measure-MatchCount -Pattern 'reason=deferred_finalize_rebuild_requested'
    rebuild_forced_dispatch             = Measure-MatchCount -Pattern 'event=REBUILD_FORCED_DISPATCH'
    policy_must_execute                 = Measure-MatchCount -Pattern 'policy=MustExecute'
    suppressed_mixed_phase_intermediate = Measure-MatchCount -Pattern 'event=REBUILD_SUPPRESSED.*reason=mixed_phase_intermediate' -Regex
}

$effectiveMetrics = [ordered]@{}
$snapshot = $null

if ($UseDeltaFromSnapshot) {
    $snapshot = Get-Snapshot -Path $SnapshotPath

    foreach ($kv in $metrics.GetEnumerator()) {
        $currentValue = [int64]$kv.Value
        $baselineValue = 0

        if ($null -ne $snapshot -and $null -ne $snapshot.metrics -and $null -ne $snapshot.metrics.$($kv.Key)) {
            $baselineValue = [int64]$snapshot.metrics.$($kv.Key)
        }

        $delta = $currentValue - $baselineValue
        if ($delta -lt 0) { $delta = 0 }
        $effectiveMetrics[$kv.Key] = $delta
    }
}
else {
    foreach ($kv in $metrics.GetEnumerator()) {
        $effectiveMetrics[$kv.Key] = [int64]$kv.Value
    }
}

$evaluation = [ordered]@{
    uiBurstEvidence           = ($effectiveMetrics.pending_duplicate -gt 0) -or ($effectiveMetrics.same_as_pending_would_merge -gt 0)
    finalizeDeferEvidence     = ($effectiveMetrics.deferred_finalize_ready -gt 0) -or ($effectiveMetrics.deferred_finalize_rebuild_req -gt 0)
    timeoutForcedDispatchSeen = ($effectiveMetrics.rebuild_forced_dispatch -gt 0)
    mustExecuteEvidence       = ($effectiveMetrics.policy_must_execute -gt 0)
}

$readyToClose81 = $evaluation.uiBurstEvidence -and
$evaluation.finalizeDeferEvidence -and
$evaluation.timeoutForcedDispatchSeen -and
$evaluation.mustExecuteEvidence

Write-Output "log=$($logItem.FullName)"
Write-Output "lastWrite=$($logItem.LastWriteTime.ToString('yyyy-MM-dd HH:mm:ss'))"
Write-Output ("mode={0}" -f ($(if ($UseDeltaFromSnapshot) { 'delta' } else { 'cumulative' })))

if ($UseDeltaFromSnapshot) {
    if ($null -eq $snapshot) {
        Write-Output "snapshot=not_found ($SnapshotPath)"
    }
    else {
        Write-Output "snapshot=loaded ($SnapshotPath)"
        if ($null -ne $snapshot.generatedAt) {
            Write-Output "snapshotGeneratedAt=$($snapshot.generatedAt)"
        }
    }
}

foreach ($kv in $effectiveMetrics.GetEnumerator()) {
    Write-Output ("{0}={1}" -f $kv.Key, $kv.Value)
}

Write-Output "--- evaluation ---"
foreach ($kv in $evaluation.GetEnumerator()) {
    Write-Output ("{0}={1}" -f $kv.Key, ($kv.Value.ToString().ToLowerInvariant()))
}

Write-Output ("readyToClose8_1={0}" -f ($readyToClose81.ToString().ToLowerInvariant()))

$guidance = Get-GenerationGuidance -ResolvedLogPath $logItem.FullName -Status 'evaluated' -Reason 'ok'
$autoCaptureResult.producedLogPath = $logItem.FullName
Write-MetricsReport -Path $ReportPath -Status 'evaluated' -Reason 'ok' -ResolvedLog $logItem.FullName -ReadyToClose $readyToClose81 -Metrics $effectiveMetrics -Evaluation $evaluation -Guidance $guidance -AutoCapture $autoCaptureResult

if ($WriteSnapshot) {
    Set-Snapshot -Path $SnapshotPath -Metrics $metrics -SourceLog $logItem.FullName -LastWrite $logItem.LastWriteTime.ToString('yyyy-MM-dd HH:mm:ss')
    Write-Output ("snapshotWritten={0}" -f $SnapshotPath)
}

if ($FailIfNotReady -and -not $readyToClose81) {
    throw "8.1 close criteria not yet satisfied in this log window."
}
