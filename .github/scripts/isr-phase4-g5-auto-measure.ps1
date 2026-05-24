param(
    [ValidateSet('run', 'compare')]
    [string]$Mode = 'run',

    [string]$ExePath = "c:\VSC_Project\ConvoPeq\build\ConvoPeq_artefacts\Release\ConvoPeq.exe",
    [string]$LogPath = "c:\VSC_Project\ConvoPeq\build\ConvoPeq_artefacts\Release\ConvoPeq.log",
    [string]$IrPath = "c:\VSC_Project\ConvoPeq\sampledata\impulse_room_correction.wav",
    [string]$OutputDir = "c:\VSC_Project\ConvoPeq\.github\tmp\phase4-g5",
    [string]$Label = "candidate",

    [string]$Order = "peq-conv",
    [string]$Phase = "mixed",
    [double]$TargetIrSec = 3.0,
    [int]$DebounceMs = 300,
    [double]$F1Hz = 160,
    [double]$F2Hz = 860,
    [double]$PreRingTau = 52,
    [int]$CliBufferSamples = 0,
    [double]$CliSampleRateHz = 0.0,
    [string]$CliDeviceType = "",
    [int]$ExitMs = 18000,
    [switch]$ProbeFinalizeAware,
    [int]$DitherBitDepth = 24,
    [int]$PostLoadDitherBitDepth = 16,
    [int]$PostLoadDelayMs = 150,
    [string]$NoiseShaper = "fixed4",
    [int]$IrReloadCount = 22,
    [int]$IrReloadIntervalMs = 260,
    [int]$BypassBurstCount = 70,
    [int]$BypassBurstIntervalMs = 30,
    [int]$BypassBurstValue = 0,
    [int]$IntentBurstCount = 55,
    [int]$IntentBurstIntervalMs = 20,

    [int]$SampleIntervalMs = 200,
    [int]$RunTimeoutMs = 120000,

    [switch]$GenerateLongIrIfMissing,
    [string]$LongIrGeneratorScript = "c:\VSC_Project\ConvoPeq\.github\scripts\generate-long-ir.ps1",

    [string]$AppendToEvidenceLog,

    [string]$BaselineJson,
    [string]$CandidateJson,
    [switch]$RequireValidFixedConfig
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function New-DirectoryIfMissing {
    param([Parameter(Mandatory = $true)][string]$Path)

    if (-not (Test-Path -LiteralPath $Path)) {
        New-Item -ItemType Directory -Path $Path -Force | Out-Null
    }
}

function Get-PercentileValue {
    param(
        [Parameter(Mandatory = $true)][double[]]$Values,
        [Parameter(Mandatory = $true)][double]$Percentile
    )

    if ($Values.Count -eq 0) { return 0.0 }
    $sorted = @($Values | Sort-Object)
    $index = [int][Math]::Ceiling(($Percentile / 100.0) * $sorted.Count) - 1
    if ($index -lt 0) { $index = 0 }
    if ($index -ge $sorted.Count) { $index = $sorted.Count - 1 }
    return [double]$sorted[$index]
}

function Get-NumericStats {
    param([double[]]$Values)

    if ($null -eq $Values -or $Values.Count -eq 0) {
        return [ordered]@{
            count = 0
            min   = 0.0
            max   = 0.0
            avg   = 0.0
            p95   = 0.0
        }
    }

    $sum = 0.0
    foreach ($v in $Values) { $sum += [double]$v }

    return [ordered]@{
        count = [int]$Values.Count
        min   = [Math]::Round(($Values | Measure-Object -Minimum).Minimum, 6)
        max   = [Math]::Round(($Values | Measure-Object -Maximum).Maximum, 6)
        avg   = [Math]::Round(($sum / $Values.Count), 6)
        p95   = [Math]::Round((Get-PercentileValue -Values $Values -Percentile 95.0), 6)
    }
}

function Measure-SimpleCount {
    param(
        [AllowEmptyCollection()][string[]]$Lines,
        [Parameter(Mandatory = $true)][string]$Pattern
    )

    return @($Lines | Where-Object { $_ -like "*$Pattern*" }).Count
}

function Measure-RegexCount {
    param(
        [AllowEmptyCollection()][string[]]$Lines,
        [Parameter(Mandatory = $true)][string]$Pattern
    )

    return @($Lines | Where-Object { $_ -match $Pattern }).Count
}

function Get-AppendedLogLines {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][int]$BeforeLineCount
    )

    if (-not (Test-Path -LiteralPath $Path)) {
        return @()
    }

    $allLines = @(Get-Content -LiteralPath $Path)
    if ($allLines.Count -lt $BeforeLineCount) {
        return @($allLines)
    }

    if ($allLines.Count -eq $BeforeLineCount) {
        return @()
    }

    return @($allLines | Select-Object -Skip $BeforeLineCount)
}

function Convert-StatsToMarkdownLine {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][hashtable]$Stats,
        [Parameter(Mandatory = $true)][string]$Unit
    )

    return ('- {0}: avg={1}{2}, p95={3}{2}, min={4}{2}, max={5}{2}, n={6}' -f $Name, $Stats.avg, $Unit, $Stats.p95, $Stats.min, $Stats.max, $Stats.count)
}

function New-RunResultObject {
    param(
        [Parameter(Mandatory = $true)][string]$RunId,
        [Parameter(Mandatory = $true)][string]$Label,
        [Parameter(Mandatory = $true)][string]$ExePath,
        [Parameter(Mandatory = $true)][string]$IrPath,
        [Parameter(Mandatory = $true)][string]$LogPath,
        [Parameter(Mandatory = $true)][int]$ExitCode,
        [Parameter(Mandatory = $true)][hashtable]$CpuStats,
        [Parameter(Mandatory = $true)][hashtable]$ProcessingTimeUsStats,
        [Parameter(Mandatory = $true)][hashtable]$LatencyStats,
        [Parameter(Mandatory = $true)][hashtable]$Telemetry,
        [Parameter(Mandatory = $true)][hashtable]$ContinuityProxy,
        [Parameter(Mandatory = $true)][hashtable]$RunConfig,
        [Parameter(Mandatory = $true)][hashtable]$Validation,
        [string[]]$AdditionalCaveats = @()
    )

    $baseCaveats = @(
        'CPU comparison is based on process CPU sampling (Get-Process CPU).',
        'Click peak (dBFS) / 20ms RMS delta (dB) cannot be computed from current logs and are out of scope for auto evaluation.',
        'mix=0 segment split is not auto-computed because log events do not currently provide explicit segment tags.'
    )

    $allCaveats = @($baseCaveats)
    if ($null -ne $AdditionalCaveats -and $AdditionalCaveats.Count -gt 0) {
        $allCaveats += $AdditionalCaveats
    }

    return [ordered]@{
        runId            = $RunId
        generatedAt      = (Get-Date).ToString('yyyy-MM-dd HH:mm:ss')
        label            = $Label
        inputs           = [ordered]@{
            exePath = $ExePath
            irPath  = $IrPath
            logPath = $LogPath
        }
        appExitCode      = $ExitCode
        cpu              = $CpuStats
        processingTimeUs = $ProcessingTimeUsStats
        rebuildLatencyMs = $LatencyStats
        telemetry        = $Telemetry
        continuityProxy  = $ContinuityProxy
        runConfig        = $RunConfig
        validation       = $Validation
        caveats          = $allCaveats
    }
}

function Get-ObjectPropertyOrDefault {
    param(
        [Parameter(Mandatory = $true)]$Object,
        [Parameter(Mandatory = $true)][string]$Name,
        $Default
    )

    if ($null -eq $Object) {
        return $Default
    }

    $property = $Object.PSObject.Properties[$Name]
    if ($null -eq $property) {
        return $Default
    }

    return $property.Value
}

function Write-RunMarkdown {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][hashtable]$RunResult
    )

    $cpu = $RunResult.cpu
    $proc = $RunResult.processingTimeUs
    $lat = $RunResult.rebuildLatencyMs
    $tel = $RunResult.telemetry
    $cont = $RunResult.continuityProxy
    $cfg = $RunResult.runConfig
    $validation = $RunResult.validation

    $lines = @()
    $lines += "# ISR Phase4 G5 Auto Measurement ($($RunResult.label))"
    $lines += ""
    $lines += "- RunId: `"$($RunResult.runId)`""
    $lines += "- GeneratedAt: $($RunResult.generatedAt)"
    $lines += "- AppExitCode: $($RunResult.appExitCode)"
    $lines += ""
    $lines += "## CPU (Process Sampling)"
    $lines += (Convert-StatsToMarkdownLine -Name 'CPU Usage' -Stats $cpu -Unit '%')
    $lines += ""
    $lines += "## Processing Time Raw (CLI_PERF_RAW)"
    $lines += (Convert-StatsToMarkdownLine -Name 'Process Time' -Stats $proc -Unit 'us')
    $lines += ""
    $lines += "## Rebuild Latency (from log latencyMs=...)"
    $lines += (Convert-StatsToMarkdownLine -Name 'Rebuild Latency' -Stats $lat -Unit 'ms')
    $lines += ""
    $lines += "## Telemetry Counters (delta in this run)"
    foreach ($kv in $tel.GetEnumerator()) {
        $lines += "- $($kv.Key): $($kv.Value)"
    }
    $lines += ""
    $lines += "## Continuity Proxy (log-based)"
    foreach ($kv in $cont.GetEnumerator()) {
        $lines += "- $($kv.Key): $($kv.Value)"
    }
    $lines += ""
    $lines += "## Fixed Audio Config Validation"
    $lines += "- Requested: deviceType=$($cfg.cliDeviceType), bufferSamples=$($cfg.cliBufferSamples), sampleRateHz=$($cfg.cliSampleRateHz)"
    $lines += "- Effective readback: bufferSamples=$($cfg.effectiveBlockSamples), sampleRateHz=$($cfg.effectiveSampleRateHz), readbackCount=$($cfg.effectiveReadbackCount)"
    $lines += "- appExitOk: $($validation.appExitOk)"
    $lines += "- bufferMatch: $($validation.bufferMatch)"
    $lines += "- sampleRateMatch: $($validation.sampleRateMatch)"
    $lines += "- fixedConfigSatisfied: $($validation.fixedConfigSatisfied)"
    $lines += ""
    $lines += "## G5 Template Paste Helper"
    $lines += "- Avg processing time (raw us): $($proc.avg)"
    $lines += "- P95 processing time (raw us): $($proc.p95)"
    $lines += "- mix=0 segment avg (Baseline / Candidate / Delta / Verdict): not auto-computed (no mix-segment tagging in logs)."
    $lines += "- mix=0 segment P95 (Baseline / Candidate / Delta / Verdict): not auto-computed (no mix-segment tagging in logs)."
    $lines += "- Click peak (dBFS): not auto-computed (no output waveform capture)."
    $lines += "- 20ms window RMS delta (dB): not auto-computed (no output waveform capture)."
    $lines += ""
    $lines += "## Caveats"
    foreach ($c in $RunResult.caveats) {
        $lines += "- $c"
    }

    Set-Content -LiteralPath $Path -Value ($lines -join [Environment]::NewLine) -Encoding UTF8
}

function Write-CompareMarkdown {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)]$Baseline,
        [Parameter(Mandatory = $true)]$Candidate,
        [Parameter(Mandatory = $true)][hashtable]$Summary
    )

    $baselineValidation = Get-ObjectPropertyOrDefault -Object $Baseline -Name 'validation' -Default ([ordered]@{ fixedConfigSatisfied = $true })
    $candidateValidation = Get-ObjectPropertyOrDefault -Object $Candidate -Name 'validation' -Default ([ordered]@{ fixedConfigSatisfied = $true })
    $baselineFixedConfigSatisfied = [bool](Get-ObjectPropertyOrDefault -Object $baselineValidation -Name 'fixedConfigSatisfied' -Default $true)
    $candidateFixedConfigSatisfied = [bool](Get-ObjectPropertyOrDefault -Object $candidateValidation -Name 'fixedConfigSatisfied' -Default $true)

    $lines = @()
    $avgVerdict = if ([double]$Summary.procTimeAvgDeltaUs -le 0.0) { 'Pass' } else { 'Fail' }
    $p95Verdict = if ([double]$Summary.procTimeP95DeltaUs -le 0.0) { 'Pass' } else { 'Fail' }
    $lines += "# ISR Phase4 G5 Auto Compare"
    $lines += ""
    $lines += "- GeneratedAt: $((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))"
    $lines += "- Baseline: `"$($Baseline.label)`" (RunId=$($Baseline.runId))"
    $lines += "- Candidate: `"$($Candidate.label)`" (RunId=$($Candidate.runId))"
    $lines += ""
    $lines += "## CPU Comparison"
    $lines += "- Avg CPU usage (Baseline / Candidate / Delta): $($Baseline.cpu.avg)% / $($Candidate.cpu.avg)% / $($Summary.cpuAvgDelta)%"
    $lines += "- P95 CPU usage (Baseline / Candidate / Delta): $($Baseline.cpu.p95)% / $($Candidate.cpu.p95)% / $($Summary.cpuP95Delta)%"
    $lines += ""
    $lines += "## Processing Time Raw Comparison"
    $lines += "- procTimeUs avg (Baseline / Candidate / Delta): $($Baseline.processingTimeUs.avg)us / $($Candidate.processingTimeUs.avg)us / $($Summary.procTimeAvgDeltaUs)us"
    $lines += "- procTimeUs P95 (Baseline / Candidate / Delta): $($Baseline.processingTimeUs.p95)us / $($Candidate.processingTimeUs.p95)us / $($Summary.procTimeP95DeltaUs)us"
    $lines += ""
    $lines += "## Fixed Audio Config Validity"
    $lines += "- baseline fixedConfigSatisfied: $baselineFixedConfigSatisfied"
    $lines += "- candidate fixedConfigSatisfied: $candidateFixedConfigSatisfied"
    $lines += "- compareDataValidity: $($baselineFixedConfigSatisfied -and $candidateFixedConfigSatisfied)"
    $lines += ""
    $lines += "## Rebuild Latency Comparison"
    $lines += "- latencyMs avg (Baseline / Candidate / Delta): $($Baseline.rebuildLatencyMs.avg)ms / $($Candidate.rebuildLatencyMs.avg)ms / $($Summary.latencyAvgDelta)ms"
    $lines += "- latencyMs P95 (Baseline / Candidate / Delta): $($Baseline.rebuildLatencyMs.p95)ms / $($Candidate.rebuildLatencyMs.p95)ms / $($Summary.latencyP95Delta)ms"
    $lines += ""
    $lines += "## G5 Template Paste Helper"
    $lines += "- Avg processing time (Baseline / Candidate / Delta / Verdict): $($Baseline.processingTimeUs.avg)us / $($Candidate.processingTimeUs.avg)us / $($Summary.procTimeAvgDeltaUs)us / $avgVerdict"
    $lines += "- P95 processing time (Baseline / Candidate / Delta / Verdict): $($Baseline.processingTimeUs.p95)us / $($Candidate.processingTimeUs.p95)us / $($Summary.procTimeP95DeltaUs)us / $p95Verdict"
    $lines += "- mix=0 segment avg (Baseline / Candidate / Delta / Verdict): not auto-computed (no mix-segment tagging in logs)."
    $lines += "- mix=0 segment P95 (Baseline / Candidate / Delta / Verdict): not auto-computed (no mix-segment tagging in logs)."
    $lines += "- Click peak (dBFS): not auto-computed (no output waveform capture)."
    $lines += "- 20ms window RMS delta (dB): not auto-computed (no output waveform capture)."

    Set-Content -LiteralPath $Path -Value ($lines -join [Environment]::NewLine) -Encoding UTF8
}

function Add-EvidenceLogSection {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][hashtable]$RunResult,
        [Parameter(Mandatory = $true)][string]$JsonPath,
        [Parameter(Mandatory = $true)][string]$MarkdownPath
    )

    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Evidence log not found: $Path"
    }

    $appendLines = @()
    $appendLines += ""
    $appendLines += "## Auto Measurement Record - $($RunResult.label) - $($RunResult.runId)"
    $appendLines += ""
    $appendLines += "- generatedAt: $($RunResult.generatedAt)"
    $appendLines += "- appExitCode: $($RunResult.appExitCode)"
    $appendLines += "- CPU avg/p95 (%): $($RunResult.cpu.avg) / $($RunResult.cpu.p95)"
    $appendLines += "- Processing raw avg/p95 (us): $($RunResult.processingTimeUs.avg) / $($RunResult.processingTimeUs.p95)"
    $appendLines += "- Rebuild latency avg/p95 (ms): $($RunResult.rebuildLatencyMs.avg) / $($RunResult.rebuildLatencyMs.p95)"
    $appendLines += "- Requested audio setup (buffer/sampleRate): $($RunResult.runConfig.cliBufferSamples) / $($RunResult.runConfig.cliSampleRateHz)"
    $appendLines += "- Effective readback (buffer/sampleRate): $($RunResult.runConfig.effectiveBlockSamples) / $($RunResult.runConfig.effectiveSampleRateHz)"
    $appendLines += "- Fixed-config validity (appExitOk/bufferMatch/sampleRateMatch/satisfied): $($RunResult.validation.appExitOk) / $($RunResult.validation.bufferMatch) / $($RunResult.validation.sampleRateMatch) / $($RunResult.validation.fixedConfigSatisfied)"
    $appendLines += "- Telemetry key count (task_queued / forced_dispatch / must_execute): $($RunResult.telemetry.task_queued) / $($RunResult.telemetry.rebuild_forced_dispatch) / $($RunResult.telemetry.policy_must_execute)"
    $appendLines += "- Continuity proxy (runtime_publish_events / fatal_like_entries): $($RunResult.continuityProxy.runtime_publish_events) / $($RunResult.continuityProxy.fatal_like_entries)"
    $appendLines += "- JSON: $JsonPath"
    $appendLines += "- Markdown: $MarkdownPath"
    $appendLines += "- Note: click peak (dBFS) and 20ms RMS delta (dB) are not auto-computed; output waveform capture is required."

    Add-Content -LiteralPath $Path -Value ($appendLines -join [Environment]::NewLine) -Encoding UTF8
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

    $baselineValidation = Get-ObjectPropertyOrDefault -Object $baseline -Name 'validation' -Default ([ordered]@{ fixedConfigSatisfied = $true })
    $candidateValidation = Get-ObjectPropertyOrDefault -Object $candidate -Name 'validation' -Default ([ordered]@{ fixedConfigSatisfied = $true })
    $baselineFixedConfigSatisfied = [bool](Get-ObjectPropertyOrDefault -Object $baselineValidation -Name 'fixedConfigSatisfied' -Default $true)
    $candidateFixedConfigSatisfied = [bool](Get-ObjectPropertyOrDefault -Object $candidateValidation -Name 'fixedConfigSatisfied' -Default $true)

    $compareDataValidity = ($baselineFixedConfigSatisfied -and $candidateFixedConfigSatisfied)

    if ($RequireValidFixedConfig -and -not $compareDataValidity) {
        throw "compare aborted: fixed-config validity check failed. baseline=$baselineFixedConfigSatisfied candidate=$candidateFixedConfigSatisfied"
    }

    $summary = [ordered]@{
        cpuAvgDelta        = [Math]::Round(([double]$candidate.cpu.avg - [double]$baseline.cpu.avg), 6)
        cpuP95Delta        = [Math]::Round(([double]$candidate.cpu.p95 - [double]$baseline.cpu.p95), 6)
        procTimeAvgDeltaUs = [Math]::Round(([double]$candidate.processingTimeUs.avg - [double]$baseline.processingTimeUs.avg), 6)
        procTimeP95DeltaUs = [Math]::Round(([double]$candidate.processingTimeUs.p95 - [double]$baseline.processingTimeUs.p95), 6)
        latencyAvgDelta    = [Math]::Round(([double]$candidate.rebuildLatencyMs.avg - [double]$baseline.rebuildLatencyMs.avg), 6)
        latencyP95Delta    = [Math]::Round(([double]$candidate.rebuildLatencyMs.p95 - [double]$baseline.rebuildLatencyMs.p95), 6)
    }

    New-DirectoryIfMissing -Path $OutputDir
    $stamp = (Get-Date).ToString('yyyyMMdd-HHmmss')
    $comparePath = Join-Path $OutputDir ("isr-phase4-g5-compare-{0}.md" -f $stamp)

    Write-CompareMarkdown -Path $comparePath -Baseline $baseline -Candidate $candidate -Summary $summary

    Write-Output "compareMarkdown=$comparePath"
    Write-Output "cpuAvgDelta=$($summary.cpuAvgDelta)"
    Write-Output "cpuP95Delta=$($summary.cpuP95Delta)"
    Write-Output "procTimeUsAvgDelta=$($summary.procTimeAvgDeltaUs)"
    Write-Output "procTimeUsP95Delta=$($summary.procTimeP95DeltaUs)"
    Write-Output "latencyAvgDeltaMs=$($summary.latencyAvgDelta)"
    Write-Output "latencyP95DeltaMs=$($summary.latencyP95Delta)"
    Write-Output "compareDataValidity=$compareDataValidity"
    exit 0
}

if (-not (Test-Path -LiteralPath $ExePath)) {
    throw "Executable not found: $ExePath"
}
if (-not (Test-Path -LiteralPath $IrPath)) {
    if ($GenerateLongIrIfMissing -and (Test-Path -LiteralPath $LongIrGeneratorScript)) {
        & $LongIrGeneratorScript -OutputPath $IrPath
    }
}
if (-not (Test-Path -LiteralPath $IrPath)) {
    throw "IR file not found: $IrPath"
}

New-DirectoryIfMissing -Path $OutputDir

$beforeLineCount = 0
if (Test-Path -LiteralPath $LogPath) {
    $beforeLineCount = [int](@(Get-Content -LiteralPath $LogPath).Count)
}

$existingProcess = Get-Process -Name "ConvoPeq" -ErrorAction SilentlyContinue
if ($null -ne $existingProcess) {
    $existingProcess | Stop-Process -Force
}

$appArgs = @(
    "--cli-run",
    "--cli-ir", $IrPath,
    "--cli-order", $Order,
    "--cli-phase", $Phase,
    "--cli-target-ir-sec", ([string]$TargetIrSec),
    "--cli-debounce-ms", ([string]$DebounceMs),
    "--cli-f1-hz", ([string]$F1Hz),
    "--cli-f2-hz", ([string]$F2Hz),
    "--cli-pre-ring-tau", ([string]$PreRingTau),
    "--cli-exit-ms", ([string]$ExitMs)
)

if ($CliBufferSamples -gt 0) {
    $appArgs += @("--cli-buffer-samples", ([string]$CliBufferSamples))
}

if ($CliSampleRateHz -gt 0.0) {
    $appArgs += @("--cli-sample-rate-hz", ([string]$CliSampleRateHz))
}

if (-not [string]::IsNullOrWhiteSpace($CliDeviceType)) {
    $appArgs += @("--cli-device-type", $CliDeviceType)
}

if ($ProbeFinalizeAware) {
    $appArgs += @(
        "--cli-noise-shaper", $NoiseShaper,
        "--cli-dither-bit-depth", ([string]$DitherBitDepth),
        "--cli-post-load-dither-bit-depth", ([string]$PostLoadDitherBitDepth),
        "--cli-post-load-delay-ms", ([string]$PostLoadDelayMs),
        "--cli-ir-reload-count", ([string]$IrReloadCount),
        "--cli-ir-reload-interval-ms", ([string]$IrReloadIntervalMs),
        "--cli-bypass-burst-count", ([string]$BypassBurstCount),
        "--cli-bypass-burst-interval-ms", ([string]$BypassBurstIntervalMs),
        "--cli-bypass-burst-value", ([string]$BypassBurstValue),
        "--cli-intent-burst-count", ([string]$IntentBurstCount),
        "--cli-intent-burst-interval-ms", ([string]$IntentBurstIntervalMs)
    )
}

$sw = [System.Diagnostics.Stopwatch]::StartNew()
$logicalCpuCount = [Environment]::ProcessorCount
$cpuSamples = New-Object System.Collections.Generic.List[double]
$lastWallSec = 0.0
$lastCpuSec = 0.0

$proc = Start-Process -FilePath $ExePath -ArgumentList $appArgs -PassThru

while (-not $proc.HasExited) {
    if ($sw.Elapsed.TotalMilliseconds -gt $RunTimeoutMs) {
        try { $proc.Kill() } catch {}
        throw "Timed out waiting for app exit (${RunTimeoutMs}ms)."
    }

    $p = Get-Process -Id $proc.Id -ErrorAction SilentlyContinue
    if ($null -ne $p) {
        $nowSec = [double]$sw.Elapsed.TotalSeconds
        $cpuSec = [double]$p.CPU

        if ($lastWallSec -gt 0.0 -and $nowSec -gt $lastWallSec) {
            $deltaCpu = $cpuSec - $lastCpuSec
            $deltaWall = $nowSec - $lastWallSec
            $cpuPct = 0.0
            if ($deltaWall -gt 0.0 -and $logicalCpuCount -gt 0) {
                $cpuPct = ($deltaCpu / $deltaWall) * 100.0 / $logicalCpuCount
            }
            if ($cpuPct -lt 0.0) { $cpuPct = 0.0 }
            $cpuSamples.Add([Math]::Round($cpuPct, 6))
        }

        $lastWallSec = $nowSec
        $lastCpuSec = $cpuSec
    }

    [System.Threading.Thread]::Sleep($SampleIntervalMs)
    $proc.Refresh()
}

$proc.Refresh()
$appExitCode = $proc.ExitCode

$appendedLines = Get-AppendedLogLines -Path $LogPath -BeforeLineCount $beforeLineCount
if ($null -eq $appendedLines) {
    $appendedLines = @()
}

$latencyValues = New-Object System.Collections.Generic.List[double]
$procTimeValues = New-Object System.Collections.Generic.List[double]
$effectiveBlockSamples = 0
$effectiveSampleRateHz = 0.0
$effectiveReadbackCount = 0
foreach ($line in $appendedLines) {
    $m = [regex]::Match($line, 'latencyMs=([0-9]+(?:\.[0-9]+)?)')
    if ($m.Success) {
        $latencyValues.Add([double]$m.Groups[1].Value)
    }

    $p = [regex]::Match($line, 'procTimeUsLast=([0-9]+(?:\.[0-9]+)?)')
    if ($p.Success) {
        $procTimeValues.Add([double]$p.Groups[1].Value)
    }

    $raw = [regex]::Match($line, 'blockSamples=([0-9]+)\s+sampleRateHz=([0-9]+(?:\.[0-9]+)?)')
    if ($raw.Success) {
        $effectiveBlockSamples = [int]$raw.Groups[1].Value
        $effectiveSampleRateHz = [double]$raw.Groups[2].Value
        $effectiveReadbackCount += 1
    }
}

$requestedAudioSetup = ($CliBufferSamples -gt 0) -or ($CliSampleRateHz -gt 0.0)
$appExitOk = ($appExitCode -eq 0)
$bufferMatch = ($CliBufferSamples -le 0) -or ($effectiveBlockSamples -eq $CliBufferSamples)
$sampleRateMatch = ($CliSampleRateHz -le 0.0) -or ([Math]::Abs($effectiveSampleRateHz - $CliSampleRateHz) -le 1.0)
$fixedConfigSatisfied = (-not $requestedAudioSetup) -or ($appExitOk -and $bufferMatch -and $sampleRateMatch)

$validation = [ordered]@{
    requestedAudioSetup  = $requestedAudioSetup
    appExitOk            = $appExitOk
    readbackAvailable    = ($effectiveReadbackCount -gt 0)
    bufferMatch          = $bufferMatch
    sampleRateMatch      = $sampleRateMatch
    fixedConfigSatisfied = $fixedConfigSatisfied
}

$additionalCaveats = @()
if ($requestedAudioSetup -and -not $fixedConfigSatisfied) {
    $additionalCaveats += ('Fixed-config validation failed. requested(buffer={0}, sampleRateHz={1}) effective(buffer={2}, sampleRateHz={3}) appExitCode={4} appExitOk={5} bufferMatch={6} sampleRateMatch={7}' -f $CliBufferSamples, $CliSampleRateHz, $effectiveBlockSamples, $effectiveSampleRateHz, $appExitCode, $appExitOk, $bufferMatch, $sampleRateMatch)
}

$telemetry = [ordered]@{
    requestRebuild_sr_bs                = Measure-SimpleCount -Lines $appendedLines -Pattern 'reason=requestRebuild_sr_bs'
    task_queued                         = Measure-SimpleCount -Lines $appendedLines -Pattern 'reason=task_queued'
    pending_duplicate                   = Measure-SimpleCount -Lines $appendedLines -Pattern 'reason=pending_duplicate'
    same_as_pending_would_merge         = Measure-SimpleCount -Lines $appendedLines -Pattern 'reason=same_as_pending_would_merge'
    deferred_finalize_ready             = Measure-SimpleCount -Lines $appendedLines -Pattern 'reason=deferred_finalize_ready'
    deferred_finalize_rebuild_req       = Measure-SimpleCount -Lines $appendedLines -Pattern 'reason=deferred_finalize_rebuild_requested'
    rebuild_forced_dispatch             = Measure-SimpleCount -Lines $appendedLines -Pattern 'event=REBUILD_FORCED_DISPATCH'
    policy_must_execute                 = Measure-SimpleCount -Lines $appendedLines -Pattern 'policy=MustExecute'
    suppressed_mixed_phase_intermediate = Measure-RegexCount -Lines $appendedLines -Pattern 'event=REBUILD_SUPPRESSED.*reason=mixed_phase_intermediate'
}

$continuityProxy = [ordered]@{
    runtime_publish_events = Measure-SimpleCount -Lines $appendedLines -Pattern '[VERIFY] runtime publish rev='
    bypass_burst_scheduled = Measure-SimpleCount -Lines $appendedLines -Pattern '[CLI] Scheduled bypass burst:'
    ir_reload_iterations   = Measure-SimpleCount -Lines $appendedLines -Pattern '[CLI] IR reload iteration='
    fatal_like_entries     = Measure-RegexCount -Lines $appendedLines -Pattern '(?i)\bfatal\b|\bassert\b|\bexception\b|\bxrun\b|\bunderrun\b|\boverrun\b'
}

$cpuStats = Get-NumericStats -Values @($cpuSamples)
$processingTimeUsStats = Get-NumericStats -Values @($procTimeValues)
$latencyStats = Get-NumericStats -Values @($latencyValues)

$runConfig = [ordered]@{
    order                  = $Order
    phase                  = $Phase
    targetIrSec            = $TargetIrSec
    debounceMs             = $DebounceMs
    f1Hz                   = $F1Hz
    f2Hz                   = $F2Hz
    preRingTau             = $PreRingTau
    cliDeviceType          = $CliDeviceType
    cliBufferSamples       = $CliBufferSamples
    cliSampleRateHz        = $CliSampleRateHz
    effectiveBlockSamples  = $effectiveBlockSamples
    effectiveSampleRateHz  = [Math]::Round($effectiveSampleRateHz, 6)
    effectiveReadbackCount = $effectiveReadbackCount
    exitMs                 = $ExitMs
    probeFinalizeAware     = [bool]$ProbeFinalizeAware
    sampleIntervalMs       = $SampleIntervalMs
    runTimeoutMs           = $RunTimeoutMs
}

$runId = (Get-Date).ToString('yyyyMMdd-HHmmss')
$result = New-RunResultObject -RunId $runId -Label $Label -ExePath $ExePath -IrPath $IrPath -LogPath $LogPath -ExitCode $appExitCode -CpuStats $cpuStats -ProcessingTimeUsStats $processingTimeUsStats -LatencyStats $latencyStats -Telemetry $telemetry -ContinuityProxy $continuityProxy -RunConfig $runConfig -Validation $validation -AdditionalCaveats $additionalCaveats

$jsonPath = Join-Path $OutputDir ("isr-phase4-g5-{0}-{1}.json" -f $Label, $runId)
$mdPath = Join-Path $OutputDir ("isr-phase4-g5-{0}-{1}.md" -f $Label, $runId)

($result | ConvertTo-Json -Depth 10) | Set-Content -LiteralPath $jsonPath -Encoding UTF8
Write-RunMarkdown -Path $mdPath -RunResult $result

if (-not [string]::IsNullOrWhiteSpace($AppendToEvidenceLog)) {
    Add-EvidenceLogSection -Path $AppendToEvidenceLog -RunResult $result -JsonPath $jsonPath -MarkdownPath $mdPath
}

Write-Output "runId=$runId"
Write-Output "json=$jsonPath"
Write-Output "markdown=$mdPath"
Write-Output "appExitCode=$appExitCode"
Write-Output "cpuAvgPct=$($cpuStats.avg)"
Write-Output "cpuP95Pct=$($cpuStats.p95)"
Write-Output "procTimeUsAvg=$($processingTimeUsStats.avg)"
Write-Output "procTimeUsP95=$($processingTimeUsStats.p95)"
Write-Output "rebuildLatencyAvgMs=$($latencyStats.avg)"
Write-Output "rebuildLatencyP95Ms=$($latencyStats.p95)"
Write-Output "fixedConfigSatisfied=$fixedConfigSatisfied"
