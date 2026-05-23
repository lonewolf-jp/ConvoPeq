param(
    [string]$LogPath = "build/ConvoPeq_artefacts/Release/ConvoPeq.log",
    [string]$SnapshotPath = ".github/tmp/isr-8_1-metrics-snapshot.json",
    [switch]$UseDeltaFromSnapshot,
    [switch]$WriteSnapshot,
    [switch]$FailIfNotReady
)

$ErrorActionPreference = 'Stop'

if (-not (Test-Path -LiteralPath $LogPath)) {
    throw "Log file not found: $LogPath"
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

if ($WriteSnapshot) {
    Set-Snapshot -Path $SnapshotPath -Metrics $metrics -SourceLog $logItem.FullName -LastWrite $logItem.LastWriteTime.ToString('yyyy-MM-dd HH:mm:ss')
    Write-Output ("snapshotWritten={0}" -f $SnapshotPath)
}

if ($FailIfNotReady -and -not $readyToClose81) {
    throw "8.1 close criteria not yet satisfied in this log window."
}
