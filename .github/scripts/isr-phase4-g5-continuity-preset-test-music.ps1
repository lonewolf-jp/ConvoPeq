param(
    [ValidateSet('autodetect', 'manual')]
    [string]$TransitionMode = 'autodetect',

    [int]$BaselineTransitionSample = -1,
    [int]$CandidateTransitionSample = -1,

    [int]$SearchStartSample = 1,
    [int]$SearchEndSample = 200000,

    [string]$BaselineLabel = 'baseline-prod',
    [string]$CandidateLabel = 'candidate-prod',

    [string]$OutputDir = 'c:\VSC_Project\ConvoPeq\.github\tmp\phase4-g5',
    [string]$AppendToEvidenceLog = 'c:\VSC_Project\ConvoPeq\doc\work\ISR_Phase4_G5_実測ログ_2026-05-23.md',

    [switch]$AllowDuplicateEvidenceLog
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$scriptRoot = Split-Path -Parent $PSCommandPath
$runner = Join-Path $scriptRoot 'isr-phase4-g5-continuity-production-rerun.ps1'
if (-not (Test-Path -LiteralPath $runner)) {
    throw "Required script not found: $runner"
}

$testMusic = 'c:\VSC_Project\ConvoPeq\sampledata\test_music.wav'
if (-not (Test-Path -LiteralPath $testMusic)) {
    throw "Preset input WAV not found: $testMusic"
}

$params = @{
    BaselineInputWav    = $testMusic
    CandidateInputWav   = $testMusic
    BaselineLabel       = $BaselineLabel
    CandidateLabel      = $CandidateLabel
    OutputDir           = $OutputDir
    AppendToEvidenceLog = $AppendToEvidenceLog
}

if ($TransitionMode -eq 'autodetect') {
    $params.BaselineAutoDetectTransition = $true
    $params.CandidateAutoDetectTransition = $true
    $params.SearchStartSample = $SearchStartSample
    $params.SearchEndSample = $SearchEndSample
}
else {
    if ($BaselineTransitionSample -lt 0 -or $CandidateTransitionSample -lt 0) {
        throw "manual mode requires -BaselineTransitionSample and -CandidateTransitionSample >= 0"
    }
    $params.BaselineTransitionSample = $BaselineTransitionSample
    $params.CandidateTransitionSample = $CandidateTransitionSample
}

if ($AllowDuplicateEvidenceLog) {
    $params.AllowDuplicateEvidenceLog = $true
}

& $runner @params
