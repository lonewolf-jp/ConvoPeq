param(
    [ValidateSet('run', 'compare')]
    [string]$Mode = 'run',

    [string]$InputWav,

    [int]$TransitionSample = -1,
    [switch]$AutoDetectTransition,
    [int]$SearchStartSample = 1,
    [int]$SearchEndSample = -1,

    [string]$Label = "candidate",
    [string]$OutputDir = "c:\VSC_Project\ConvoPeq\.github\tmp\phase4-g5",
    [int]$PeakWindowSamples = 256,
    [double]$RmsWindowMs = 20.0,
    [double]$ClickThresholdDbfs = -70.0,
    [double]$RmsDeltaThresholdDb = 1.5,
    [string]$AppendToEvidenceLog,
    [switch]$AllowDuplicateEvidenceLog,

    [string]$BaselineJson,
    [string]$CandidateJson
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function New-DirectoryIfMissing {
    param([Parameter(Mandatory = $true)][string]$Path)

    if (-not (Test-Path -LiteralPath $Path)) {
        New-Item -ItemType Directory -Path $Path -Force | Out-Null
    }
}

function Convert-ToDbfs {
    param([double]$Value)

    if ($Value -le 1.0e-12) {
        return -160.0
    }

    return [Math]::Round((20.0 * [Math]::Log10($Value)), 6)
}

function Read-Pcm24AsDouble {
    param([Parameter(Mandatory = $true)][System.IO.BinaryReader]$Reader)

    $b0 = [int]$Reader.ReadByte()
    $b1 = [int]$Reader.ReadByte()
    $b2 = [int]$Reader.ReadByte()

    $raw = ($b0 -bor ($b1 -shl 8) -bor ($b2 -shl 16))
    if ($raw -ge 0x800000) {
        $raw -= 0x1000000
    }

    return ([double]$raw / 8388608.0)
}

function Read-WavHeader {
    param([Parameter(Mandatory = $true)][string]$Path)

    $fs = [System.IO.File]::Open($Path, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read, [System.IO.FileShare]::Read)
    try {
        $br = New-Object System.IO.BinaryReader($fs)
        try {
            $riff = [System.Text.Encoding]::ASCII.GetString($br.ReadBytes(4))
            if ($riff -ne 'RIFF') {
                throw "Invalid WAV: missing RIFF header"
            }

            [void]$br.ReadInt32() # file size

            $wave = [System.Text.Encoding]::ASCII.GetString($br.ReadBytes(4))
            if ($wave -ne 'WAVE') {
                throw "Invalid WAV: missing WAVE signature"
            }

            $fmtFound = $false
            $dataFound = $false
            $formatTag = 0
            $channels = 0
            $sampleRate = 0
            $bitsPerSample = 0
            $dataOffset = 0
            $dataSize = 0

            while ($fs.Position -lt $fs.Length) {
                if (($fs.Length - $fs.Position) -lt 8) {
                    break
                }

                $chunkId = [System.Text.Encoding]::ASCII.GetString($br.ReadBytes(4))
                $chunkSize = $br.ReadInt32()
                $chunkDataPos = $fs.Position

                if ($chunkId -eq 'fmt ') {
                    $fmtFound = $true
                    $formatTag = $br.ReadInt16()
                    $channels = $br.ReadInt16()
                    $sampleRate = $br.ReadInt32()
                    [void]$br.ReadInt32() # byteRate
                    [void]$br.ReadInt16() # blockAlign
                    $bitsPerSample = $br.ReadInt16()
                }
                elseif ($chunkId -eq 'data') {
                    $dataFound = $true
                    $dataOffset = $fs.Position
                    $dataSize = $chunkSize
                    break
                }

                $nextPos = $chunkDataPos + $chunkSize
                if (($chunkSize % 2) -eq 1) {
                    $nextPos += 1
                }

                if ($nextPos -gt $fs.Length) {
                    break
                }
                $fs.Position = $nextPos
            }

            if (-not $fmtFound) {
                throw "Invalid WAV: fmt chunk not found"
            }
            if (-not $dataFound) {
                throw "Invalid WAV: data chunk not found"
            }
            if (-not (($formatTag -eq 1 -and ($bitsPerSample -eq 16 -or $bitsPerSample -eq 24)) -or ($formatTag -eq 3 -and $bitsPerSample -eq 32))) {
                throw "Unsupported WAV formatTag/bits: formatTag=$formatTag bitsPerSample=$bitsPerSample. Supported: PCM16/PCM24 (1/16,1/24), IEEE float32 (3/32)."
            }
            if ($channels -lt 1 -or $channels -gt 2) {
                throw "Unsupported channel count ($channels). Only mono/stereo are supported."
            }

            return [ordered]@{
                channels      = [int]$channels
                sampleRate    = [int]$sampleRate
                formatTag     = [int]$formatTag
                bitsPerSample = [int]$bitsPerSample
                dataOffset    = [int64]$dataOffset
                dataSize      = [int64]$dataSize
            }
        }
        finally {
            $br.Dispose()
        }
    }
    finally {
        $fs.Dispose()
    }
}

function Measure-ContinuityFromWav {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][int]$TransitionFrame,
        [Parameter(Mandatory = $true)][int]$PeakWindowFrames,
        [Parameter(Mandatory = $true)][double]$RmsWindowMs
    )

    $header = Read-WavHeader -Path $Path
    $channels = [int]$header.channels
    $sampleRate = [int]$header.sampleRate
    $formatTag = [int]$header.formatTag
    $bitsPerSample = [int]$header.bitsPerSample
    $dataOffset = [int64]$header.dataOffset
    $dataSize = [int64]$header.dataSize

    $bytesPerSample = [int]($bitsPerSample / 8)
    $bytesPerFrame = $channels * $bytesPerSample
    if ($bytesPerFrame -le 0) {
        throw "Invalid WAV frame size"
    }

    $totalFrames = [int]([Math]::Floor($dataSize / $bytesPerFrame))
    if ($totalFrames -le 0) {
        throw "WAV has no audio frames"
    }

    if ($TransitionFrame -lt 0 -or $TransitionFrame -ge $totalFrames) {
        throw "TransitionSample out of range. transition=$TransitionFrame totalFrames=$totalFrames"
    }

    $rmsWindowFrames = [int][Math]::Round(($RmsWindowMs / 1000.0) * $sampleRate)
    if ($rmsWindowFrames -lt 1) {
        $rmsWindowFrames = 1
    }

    $peakStart = [Math]::Max(0, $TransitionFrame - $PeakWindowFrames)
    $peakEnd = [Math]::Min($totalFrames - 1, $TransitionFrame + $PeakWindowFrames)

    $preStart = [Math]::Max(0, $TransitionFrame - $rmsWindowFrames)
    $preEnd = [Math]::Max($preStart, $TransitionFrame - 1)

    $postStart = [Math]::Min($totalFrames - 1, $TransitionFrame)
    $postEnd = [Math]::Min($totalFrames - 1, $postStart + $rmsWindowFrames - 1)

    $rangeStart = [Math]::Min($peakStart, [Math]::Min($preStart, $postStart))
    $rangeEnd = [Math]::Max($peakEnd, [Math]::Max($preEnd, $postEnd))

    $peakAbs = 0.0
    $preEnergy = 0.0
    $postEnergy = 0.0
    $preCount = 0
    $postCount = 0

    $fs = [System.IO.File]::Open($Path, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read, [System.IO.FileShare]::Read)
    try {
        $fs.Position = $dataOffset + ([int64]$rangeStart * [int64]$bytesPerFrame)
        $br = New-Object System.IO.BinaryReader($fs)
        try {
            for ($frame = $rangeStart; $frame -le $rangeEnd; $frame++) {
                for ($ch = 0; $ch -lt $channels; $ch++) {
                    $v = 0.0
                    if ($formatTag -eq 1 -and $bitsPerSample -eq 16) {
                        $s16 = $br.ReadInt16()
                        $v = [double]$s16 / 32768.0
                    }
                    elseif ($formatTag -eq 1 -and $bitsPerSample -eq 24) {
                        $v = Read-Pcm24AsDouble -Reader $br
                    }
                    elseif ($formatTag -eq 3 -and $bitsPerSample -eq 32) {
                        $f32 = $br.ReadSingle()
                        $v = [double]$f32
                    }
                    else {
                        throw "Unsupported sample decoder path for formatTag=$formatTag bitsPerSample=$bitsPerSample"
                    }

                    $absV = [Math]::Abs($v)

                    if ($frame -ge $peakStart -and $frame -le $peakEnd) {
                        if ($absV -gt $peakAbs) {
                            $peakAbs = $absV
                        }
                    }

                    if ($frame -ge $preStart -and $frame -le $preEnd) {
                        $preEnergy += ($v * $v)
                        $preCount++
                    }

                    if ($frame -ge $postStart -and $frame -le $postEnd) {
                        $postEnergy += ($v * $v)
                        $postCount++
                    }
                }
            }
        }
        finally {
            $br.Dispose()
        }
    }
    finally {
        $fs.Dispose()
    }

    $preRms = if ($preCount -gt 0) { [Math]::Sqrt($preEnergy / $preCount) } else { 0.0 }
    $postRms = if ($postCount -gt 0) { [Math]::Sqrt($postEnergy / $postCount) } else { 0.0 }

    $clickPeakDbfs = Convert-ToDbfs -Value $peakAbs
    $preRmsDbfs = Convert-ToDbfs -Value $preRms
    $postRmsDbfs = Convert-ToDbfs -Value $postRms
    $rmsDeltaDb = [Math]::Round([Math]::Abs($postRmsDbfs - $preRmsDbfs), 6)

    return [ordered]@{
        channels         = $channels
        sampleRate       = $sampleRate
        totalFrames      = $totalFrames
        transitionFrame  = $TransitionFrame
        peakWindowFrames = $PeakWindowFrames
        rmsWindowMs      = $RmsWindowMs
        clickPeakDbfs    = $clickPeakDbfs
        preRmsDbfs       = $preRmsDbfs
        postRmsDbfs      = $postRmsDbfs
        rmsDeltaDb       = $rmsDeltaDb
    }
}

function Find-TransitionFromWav {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [int]$SearchStartFrame = 1,
        [int]$SearchEndFrame = -1
    )

    $header = Read-WavHeader -Path $Path
    $channels = [int]$header.channels
    $formatTag = [int]$header.formatTag
    $bitsPerSample = [int]$header.bitsPerSample
    $dataOffset = [int64]$header.dataOffset
    $dataSize = [int64]$header.dataSize

    $bytesPerSample = [int]($bitsPerSample / 8)
    $bytesPerFrame = $channels * $bytesPerSample
    $totalFrames = [int]([Math]::Floor($dataSize / $bytesPerFrame))
    if ($totalFrames -lt 2) {
        throw "WAV has too few frames for transition detection"
    }

    if ($SearchStartFrame -lt 1) { $SearchStartFrame = 1 }
    if ($SearchEndFrame -lt 0 -or $SearchEndFrame -ge $totalFrames) {
        $SearchEndFrame = $totalFrames - 1
    }
    if ($SearchStartFrame -gt $SearchEndFrame) {
        throw "Invalid search range: start=$SearchStartFrame end=$SearchEndFrame total=$totalFrames"
    }

    $maxDiff = -1.0
    $bestFrame = $SearchStartFrame
    $readStartFrame = [Math]::Max(0, $SearchStartFrame - 1)

    $fs = [System.IO.File]::Open($Path, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read, [System.IO.FileShare]::Read)
    try {
        $fs.Position = $dataOffset + ([int64]$readStartFrame * [int64]$bytesPerFrame)
        $br = New-Object System.IO.BinaryReader($fs)
        try {
            $prevAbs = 0.0
            for ($frame = $readStartFrame; $frame -le $SearchEndFrame; $frame++) {
                $frameAbs = 0.0
                for ($ch = 0; $ch -lt $channels; $ch++) {
                    $v = 0.0
                    if ($formatTag -eq 1 -and $bitsPerSample -eq 16) {
                        $v = [double]$br.ReadInt16() / 32768.0
                    }
                    elseif ($formatTag -eq 1 -and $bitsPerSample -eq 24) {
                        $v = Read-Pcm24AsDouble -Reader $br
                    }
                    elseif ($formatTag -eq 3 -and $bitsPerSample -eq 32) {
                        $v = [double]$br.ReadSingle()
                    }
                    else {
                        throw "Unsupported sample decoder path for formatTag=$formatTag bitsPerSample=$bitsPerSample"
                    }

                    $a = [Math]::Abs($v)
                    if ($a -gt $frameAbs) { $frameAbs = $a }
                }

                if ($frame -ge $SearchStartFrame -and $frame -le $SearchEndFrame) {
                    $diff = [Math]::Abs($frameAbs - $prevAbs)
                    if ($diff -gt $maxDiff) {
                        $maxDiff = $diff
                        $bestFrame = $frame
                    }
                }

                $prevAbs = $frameAbs
            }
        }
        finally {
            $br.Dispose()
        }
    }
    finally {
        $fs.Dispose()
    }

    return [ordered]@{
        transitionFrame  = $bestFrame
        score            = [Math]::Round($maxDiff, 9)
        searchStartFrame = $SearchStartFrame
        searchEndFrame   = $SearchEndFrame
        totalFrames      = $totalFrames
    }
}

function Write-ContinuityCompareMarkdown {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)]$Baseline,
        [Parameter(Mandatory = $true)]$Candidate,
        [Parameter(Mandatory = $true)][hashtable]$Summary
    )

    $lines = @()
    $lines += "# ISR Phase4 G5 Continuity Compare"
    $lines += ""
    $lines += "- GeneratedAt: $((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))"
    $lines += "- Baseline: `"$($Baseline.label)`" (RunId=$($Baseline.runId))"
    $lines += "- Candidate: `"$($Candidate.label)`" (RunId=$($Candidate.runId))"
    $lines += ""
    $lines += "## Continuity Metrics"
    $lines += "- Click peak (dBFS) Baseline / Candidate / Delta(C-B): $($Baseline.metrics.clickPeakDbfs) / $($Candidate.metrics.clickPeakDbfs) / $($Summary.clickPeakDeltaDbfs)"
    $lines += "- 20ms RMS delta (dB) Baseline / Candidate / Delta(C-B): $($Baseline.metrics.rmsDeltaDb) / $($Candidate.metrics.rmsDeltaDb) / $($Summary.rmsDeltaDeltaDb)"
    $lines += ""
    $lines += "## Verdict"
    $lines += "- Baseline overallPass: $($Baseline.verdict.overallPass)"
    $lines += "- Candidate overallPass: $($Candidate.verdict.overallPass)"
    $lines += ""
    $lines += "## Paste Helper"
    $lines += "- Click peak (dBFS): Baseline=$($Baseline.metrics.clickPeakDbfs), Candidate=$($Candidate.metrics.clickPeakDbfs), Delta(C-B)=$($Summary.clickPeakDeltaDbfs)"
    $lines += "- 20ms RMS delta (dB): Baseline=$($Baseline.metrics.rmsDeltaDb), Candidate=$($Candidate.metrics.rmsDeltaDb), Delta(C-B)=$($Summary.rmsDeltaDeltaDb)"

    Set-Content -LiteralPath $Path -Value ($lines -join [Environment]::NewLine) -Encoding UTF8
}

function Test-ExistingCompareRecord {
    param(
        [Parameter(Mandatory = $true)][string]$EvidenceLogPath,
        [Parameter(Mandatory = $true)][string]$BaselineJsonPath,
        [Parameter(Mandatory = $true)][string]$CandidateJsonPath,
        [Parameter(Mandatory = $true)][double]$ClickPeakDelta,
        [Parameter(Mandatory = $true)][double]$RmsDeltaDelta
    )

    if (-not (Test-Path -LiteralPath $EvidenceLogPath)) {
        return $false
    }

    $raw = Get-Content -LiteralPath $EvidenceLogPath -Raw
    if ([string]::IsNullOrWhiteSpace($raw)) {
        return $false
    }

    $clickText = [string]::Format([System.Globalization.CultureInfo]::InvariantCulture, "{0:0.######}", $ClickPeakDelta)
    $rmsText = [string]::Format([System.Globalization.CultureInfo]::InvariantCulture, "{0:0.######}", $RmsDeltaDelta)

    return (
        $raw.Contains("- baselineJson: $BaselineJsonPath") -and
        $raw.Contains("- candidateJson: $CandidateJsonPath") -and
        $raw.Contains("- clickPeakDelta(C-B,dBFS): $clickText") -and
        $raw.Contains("- rmsDelta20msDelta(C-B,dB): $rmsText")
    )
}

if ($Mode -eq 'compare') {
    if ([string]::IsNullOrWhiteSpace($BaselineJson) -or [string]::IsNullOrWhiteSpace($CandidateJson)) {
        throw "compare mode requires -BaselineJson and -CandidateJson"
    }
    if (-not (Test-Path -LiteralPath $BaselineJson)) {
        throw "Baseline json not found: $BaselineJson"
    }
    if (-not (Test-Path -LiteralPath $CandidateJson)) {
        throw "Candidate json not found: $CandidateJson"
    }

    $baseline = (Get-Content -LiteralPath $BaselineJson -Raw | ConvertFrom-Json)
    $candidate = (Get-Content -LiteralPath $CandidateJson -Raw | ConvertFrom-Json)

    $summary = [ordered]@{
        clickPeakDeltaDbfs = [Math]::Round(([double]$candidate.metrics.clickPeakDbfs - [double]$baseline.metrics.clickPeakDbfs), 6)
        rmsDeltaDeltaDb    = [Math]::Round(([double]$candidate.metrics.rmsDeltaDb - [double]$baseline.metrics.rmsDeltaDb), 6)
    }

    New-DirectoryIfMissing -Path $OutputDir
    $stamp = (Get-Date).ToString('yyyyMMdd-HHmmss')
    $compareJsonPath = Join-Path $OutputDir ("isr-phase4-g5-continuity-compare-{0}.json" -f $stamp)
    $comparePath = Join-Path $OutputDir ("isr-phase4-g5-continuity-compare-{0}.md" -f $stamp)

    $compareResult = [ordered]@{
        runId       = $stamp
        generatedAt = (Get-Date).ToString('yyyy-MM-dd HH:mm:ss')
        baseline    = [ordered]@{
            jsonPath      = $BaselineJson
            runId         = $baseline.runId
            label         = $baseline.label
            clickPeakDbfs = $baseline.metrics.clickPeakDbfs
            rmsDeltaDb    = $baseline.metrics.rmsDeltaDb
            overallPass   = $baseline.verdict.overallPass
        }
        candidate   = [ordered]@{
            jsonPath      = $CandidateJson
            runId         = $candidate.runId
            label         = $candidate.label
            clickPeakDbfs = $candidate.metrics.clickPeakDbfs
            rmsDeltaDb    = $candidate.metrics.rmsDeltaDb
            overallPass   = $candidate.verdict.overallPass
        }
        summary     = $summary
    }

    ($compareResult | ConvertTo-Json -Depth 10) | Set-Content -LiteralPath $compareJsonPath -Encoding UTF8
    Write-ContinuityCompareMarkdown -Path $comparePath -Baseline $baseline -Candidate $candidate -Summary $summary

    if (-not [string]::IsNullOrWhiteSpace($AppendToEvidenceLog)) {
        if (-not (Test-Path -LiteralPath $AppendToEvidenceLog)) {
            throw "Evidence log not found: $AppendToEvidenceLog"
        }

        $exists = Test-ExistingCompareRecord -EvidenceLogPath $AppendToEvidenceLog -BaselineJsonPath $BaselineJson -CandidateJsonPath $CandidateJson -ClickPeakDelta ([double]$summary.clickPeakDeltaDbfs) -RmsDeltaDelta ([double]$summary.rmsDeltaDeltaDb)
        if (-not $exists -or $AllowDuplicateEvidenceLog) {
            $append = @()
            $append += ""
            $append += "## Auto Continuity Compare Record - $($baseline.label)-vs-$($candidate.label) - $stamp"
            $append += ""
            $append += "- baselineJson: $BaselineJson"
            $append += "- candidateJson: $CandidateJson"
            $append += "- clickPeakDelta(C-B,dBFS): $($summary.clickPeakDeltaDbfs)"
            $append += "- rmsDelta20msDelta(C-B,dB): $($summary.rmsDeltaDeltaDb)"
            $append += "- Compare JSON: $compareJsonPath"
            $append += "- Compare Markdown: $comparePath"

            Add-Content -LiteralPath $AppendToEvidenceLog -Value ($append -join [Environment]::NewLine) -Encoding UTF8
            Write-Output "evidenceAppend=added"
        }
        else {
            Write-Output "evidenceAppend=skipped-duplicate"
        }
    }

    Write-Output "compareJson=$compareJsonPath"
    Write-Output "compareMarkdown=$comparePath"
    Write-Output "clickPeakDeltaDbfs=$($summary.clickPeakDeltaDbfs)"
    Write-Output "rmsDeltaDeltaDb=$($summary.rmsDeltaDeltaDb)"
    exit 0
}

if (-not (Test-Path -LiteralPath $InputWav)) {
    throw "Input WAV not found: $InputWav"
}

if ($PeakWindowSamples -lt 1) {
    throw "PeakWindowSamples must be >= 1"
}
if ($RmsWindowMs -le 0.0) {
    throw "RmsWindowMs must be > 0"
}

if (-not $AutoDetectTransition -and $TransitionSample -lt 0) {
    throw "Specify -TransitionSample >= 0, or use -AutoDetectTransition."
}

New-DirectoryIfMissing -Path $OutputDir

$autoDetected = $null
if ($AutoDetectTransition) {
    $autoDetected = Find-TransitionFromWav -Path $InputWav -SearchStartFrame $SearchStartSample -SearchEndFrame $SearchEndSample
    $TransitionSample = [int]$autoDetected.transitionFrame
}

$metrics = Measure-ContinuityFromWav -Path $InputWav -TransitionFrame $TransitionSample -PeakWindowFrames $PeakWindowSamples -RmsWindowMs $RmsWindowMs

$clickPass = ($metrics.clickPeakDbfs -le $ClickThresholdDbfs)
$rmsPass = ($metrics.rmsDeltaDb -le $RmsDeltaThresholdDb)
$overallPass = ($clickPass -and $rmsPass)

$runId = (Get-Date).ToString('yyyyMMdd-HHmmss')
$result = [ordered]@{
    runId               = $runId
    generatedAt         = (Get-Date).ToString('yyyy-MM-dd HH:mm:ss')
    label               = $Label
    inputWav            = $InputWav
    transitionSelection = [ordered]@{
        autoDetected     = [bool]$AutoDetectTransition
        transitionSample = $TransitionSample
        detection        = $autoDetected
    }
    thresholds          = [ordered]@{
        clickThresholdDbfs  = $ClickThresholdDbfs
        rmsDeltaThresholdDb = $RmsDeltaThresholdDb
    }
    metrics             = $metrics
    verdict             = [ordered]@{
        clickPass   = $clickPass
        rmsPass     = $rmsPass
        overallPass = $overallPass
    }
}

$jsonPath = Join-Path $OutputDir ("isr-phase4-g5-continuity-{0}-{1}.json" -f $Label, $runId)
$mdPath = Join-Path $OutputDir ("isr-phase4-g5-continuity-{0}-{1}.md" -f $Label, $runId)

($result | ConvertTo-Json -Depth 10) | Set-Content -LiteralPath $jsonPath -Encoding UTF8

$lines = @()
$lines += "# ISR Phase4 G5 Continuity Metrics ($Label)"
$lines += ""
$lines += "- RunId: $runId"
$lines += "- GeneratedAt: $($result.generatedAt)"
$lines += "- InputWav: $InputWav"
$lines += "- TransitionSample: $TransitionSample"
if ($AutoDetectTransition -and $null -ne $autoDetected) {
    $lines += "- TransitionAutoDetectScore: $($autoDetected.score)"
    $lines += "- SearchRange: [$($autoDetected.searchStartFrame), $($autoDetected.searchEndFrame)]"
}
$lines += ""
$lines += "## Metrics"
$lines += "- Click peak (dBFS): $($metrics.clickPeakDbfs)"
$lines += "- Pre RMS (dBFS): $($metrics.preRmsDbfs)"
$lines += "- Post RMS (dBFS): $($metrics.postRmsDbfs)"
$lines += "- 20ms RMS delta (dB): $($metrics.rmsDeltaDb)"
$lines += ""
$lines += "## Thresholds"
$lines += "- Click threshold (dBFS): <= $ClickThresholdDbfs"
$lines += "- RMS delta threshold (dB): <= $RmsDeltaThresholdDb"
$lines += ""
$lines += "## Verdict"
$lines += "- clickPass: $($clickPass.ToString().ToLowerInvariant())"
$lines += "- rmsPass: $($rmsPass.ToString().ToLowerInvariant())"
$lines += "- overallPass: $($overallPass.ToString().ToLowerInvariant())"

Set-Content -LiteralPath $mdPath -Value ($lines -join [Environment]::NewLine) -Encoding UTF8

if (-not [string]::IsNullOrWhiteSpace($AppendToEvidenceLog)) {
    if (-not (Test-Path -LiteralPath $AppendToEvidenceLog)) {
        throw "Evidence log not found: $AppendToEvidenceLog"
    }

    $append = @()
    $append += ""
    $append += "## Auto Continuity Record - $Label - $runId"
    $append += ""
    $append += "- inputWav: $InputWav"
    $append += "- transitionSample: $TransitionSample"
    if ($AutoDetectTransition -and $null -ne $autoDetected) {
        $append += "- transitionAutoDetectScore: $($autoDetected.score)"
        $append += "- transitionSearchRange: [$($autoDetected.searchStartFrame), $($autoDetected.searchEndFrame)]"
    }
    $append += "- clickPeak(dBFS): $($metrics.clickPeakDbfs)"
    $append += "- rmsDelta20ms(dB): $($metrics.rmsDeltaDb)"
    $append += "- verdict(overall): $($overallPass.ToString().ToLowerInvariant())"
    $append += "- JSON: $jsonPath"
    $append += "- Markdown: $mdPath"

    Add-Content -LiteralPath $AppendToEvidenceLog -Value ($append -join [Environment]::NewLine) -Encoding UTF8
}

Write-Output "runId=$runId"
Write-Output "json=$jsonPath"
Write-Output "markdown=$mdPath"
Write-Output "clickPeakDbfs=$($metrics.clickPeakDbfs)"
Write-Output "rmsDeltaDb=$($metrics.rmsDeltaDb)"
Write-Output "overallPass=$($overallPass.ToString().ToLowerInvariant())"
