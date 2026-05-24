param(
    [Parameter(Mandatory = $true)]
    [string]$BaselineInputWav,

    [Parameter(Mandatory = $true)]
    [string]$CandidateInputWav,

    [string]$BaselineLabel = "baseline-prod",
    [string]$CandidateLabel = "candidate-prod",

    [int]$BaselineTransitionSample = -1,
    [int]$CandidateTransitionSample = -1,

    [switch]$BaselineAutoDetectTransition,
    [switch]$CandidateAutoDetectTransition,

    [int]$SearchStartSample = 1,
    [int]$SearchEndSample = -1,

    [string]$OutputDir = "c:\VSC_Project\ConvoPeq\.github\tmp\phase4-g5",
    [int]$PeakWindowSamples = 256,
    [double]$RmsWindowMs = 20.0,
    [double]$ClickThresholdDbfs = -70.0,
    [double]$RmsDeltaThresholdDb = 1.5,

    [string]$AppendToEvidenceLog,
    [switch]$AllowDuplicateEvidenceLog
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$scriptRoot = Split-Path -Parent $PSCommandPath
$continuityScript = Join-Path $scriptRoot "isr-phase4-g5-continuity-metrics.ps1"
if (-not (Test-Path -LiteralPath $continuityScript)) {
    throw "Required script not found: $continuityScript"
}

function Get-KeyValueMap {
    param([Parameter(Mandatory = $true)][object[]]$Lines)

    $map = @{}
    foreach ($line in $Lines) {
        if ($line -is [string] -and $line -match '^([^=]+)=(.*)$') {
            $map[$matches[1]] = $matches[2]
        }
    }
    return $map
}

function Invoke-ContinuityRun {
    param(
        [Parameter(Mandatory = $true)][string]$InputWav,
        [Parameter(Mandatory = $true)][string]$Label,
        [Parameter(Mandatory = $true)][int]$TransitionSample,
        [Parameter(Mandatory = $true)][bool]$AutoDetect,
        [string]$AppendLog
    )

    if (-not (Test-Path -LiteralPath $InputWav)) {
        throw "Input WAV not found: $InputWav"
    }

    $runParams = @{
        Mode                = 'run'
        InputWav            = $InputWav
        Label               = $Label
        OutputDir           = $OutputDir
        PeakWindowSamples   = $PeakWindowSamples
        RmsWindowMs         = $RmsWindowMs
        ClickThresholdDbfs  = $ClickThresholdDbfs
        RmsDeltaThresholdDb = $RmsDeltaThresholdDb
    }

    if ($AutoDetect) {
        $runParams.AutoDetectTransition = $true
        $runParams.SearchStartSample = $SearchStartSample
        $runParams.SearchEndSample = $SearchEndSample
    }
    else {
        if ($TransitionSample -lt 0) {
            throw "When auto-detect is off, TransitionSample must be >= 0 for label '$Label'"
        }
        $runParams.TransitionSample = $TransitionSample
    }

    if (-not [string]::IsNullOrWhiteSpace($AppendLog)) {
        $runParams.AppendToEvidenceLog = $AppendLog
    }

    $raw = & $continuityScript @runParams
    $kv = Get-KeyValueMap -Lines $raw
    if (-not $kv.ContainsKey('json')) {
        throw "Run mode did not produce json output for label '$Label'"
    }

    return [ordered]@{
        label         = $Label
        json          = $kv['json']
        markdown      = if ($kv.ContainsKey('markdown')) { $kv['markdown'] } else { $null }
        runId         = if ($kv.ContainsKey('runId')) { $kv['runId'] } else { $null }
        clickPeakDbfs = if ($kv.ContainsKey('clickPeakDbfs')) { $kv['clickPeakDbfs'] } else { $null }
        rmsDeltaDb    = if ($kv.ContainsKey('rmsDeltaDb')) { $kv['rmsDeltaDb'] } else { $null }
        overallPass   = if ($kv.ContainsKey('overallPass')) { $kv['overallPass'] } else { $null }
    }
}

$baseline = Invoke-ContinuityRun -InputWav $BaselineInputWav -Label $BaselineLabel -TransitionSample $BaselineTransitionSample -AutoDetect ([bool]$BaselineAutoDetectTransition) -AppendLog $AppendToEvidenceLog
$candidate = Invoke-ContinuityRun -InputWav $CandidateInputWav -Label $CandidateLabel -TransitionSample $CandidateTransitionSample -AutoDetect ([bool]$CandidateAutoDetectTransition) -AppendLog $AppendToEvidenceLog

$compareParams = @{
    Mode          = 'compare'
    BaselineJson  = $baseline.json
    CandidateJson = $candidate.json
    OutputDir     = $OutputDir
}
if (-not [string]::IsNullOrWhiteSpace($AppendToEvidenceLog)) {
    $compareParams.AppendToEvidenceLog = $AppendToEvidenceLog
}
if ($AllowDuplicateEvidenceLog) {
    $compareParams.AllowDuplicateEvidenceLog = $true
}

$compareRaw = & $continuityScript @compareParams
$compareKv = Get-KeyValueMap -Lines $compareRaw

Write-Output "baselineJson=$($baseline.json)"
Write-Output "candidateJson=$($candidate.json)"
if ($compareKv.ContainsKey('compareJson')) {
    Write-Output "compareJson=$($compareKv['compareJson'])"
}
if ($compareKv.ContainsKey('compareMarkdown')) {
    Write-Output "compareMarkdown=$($compareKv['compareMarkdown'])"
}
if ($compareKv.ContainsKey('clickPeakDeltaDbfs')) {
    Write-Output "clickPeakDeltaDbfs=$($compareKv['clickPeakDeltaDbfs'])"
}
if ($compareKv.ContainsKey('rmsDeltaDeltaDb')) {
    Write-Output "rmsDeltaDeltaDb=$($compareKv['rmsDeltaDeltaDb'])"
}
if ($compareKv.ContainsKey('evidenceAppend')) {
    Write-Output "evidenceAppend=$($compareKv['evidenceAppend'])"
}
