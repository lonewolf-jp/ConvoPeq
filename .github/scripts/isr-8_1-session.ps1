param(
    [Parameter(ParameterSetName = 'Begin', Mandatory = $true)]
    [switch]$Begin,

    [Parameter(ParameterSetName = 'End', Mandatory = $true)]
    [switch]$End,

    [Parameter(ParameterSetName = 'Status', Mandatory = $true)]
    [switch]$Status,

    [string]$LogPath = "c:\VSC_Project\ConvoPeq\build\ConvoPeq_artefacts\Release\ConvoPeq.log",
    [string]$SnapshotPath = "c:\VSC_Project\ConvoPeq\.github\tmp\isr-8_1-metrics-snapshot.json"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$metricsScript = "c:\VSC_Project\ConvoPeq\.github\scripts\isr-rebuild-admission-8_1-metrics.ps1"
if (-not (Test-Path $metricsScript)) {
    throw "Metrics script not found: $metricsScript"
}
if (-not (Test-Path $LogPath)) {
    throw "Log file not found: $LogPath"
}

if ($Begin) {
    Write-Output "[ISR-8.1] Begin session: write snapshot"
    & $metricsScript -LogPath $LogPath -SnapshotPath $SnapshotPath -WriteSnapshot
    Write-Output "[ISR-8.1] Next: perform UI operations, then run: .\\.github\\scripts\\isr-8_1-session.ps1 -End"
    exit 0
}

if ($End) {
    Write-Output "[ISR-8.1] End session: delta evaluation"
    if (-not (Test-Path $SnapshotPath)) {
        throw "Snapshot not found: $SnapshotPath. Run -Begin first."
    }
    & $metricsScript -LogPath $LogPath -SnapshotPath $SnapshotPath -UseDeltaFromSnapshot
    exit 0
}

if ($Status) {
    Write-Output "[ISR-8.1] Current cumulative status"
    & $metricsScript -LogPath $LogPath
    exit 0
}
