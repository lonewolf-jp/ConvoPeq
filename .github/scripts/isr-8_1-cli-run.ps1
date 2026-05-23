param(
    [switch]$BeginOnly,
    [switch]$EndOnly,

    [string]$SessionScriptPath = "c:\VSC_Project\ConvoPeq\.github\scripts\isr-8_1-session.ps1",
    [string]$ExePath = "c:\VSC_Project\ConvoPeq\build\ConvoPeq_artefacts\Release\ConvoPeq.exe",
    [string]$IrPath = "c:\VSC_Project\ConvoPeq\sampledata\synthetic_long_ir_20s.wav",

    [string]$Order = "conv-peq",
    [string]$Phase = "mixed",
    [double]$TargetIrSec = 2.4,
    [int]$DebounceMs = 550,
    [double]$F1Hz = 210,
    [double]$F2Hz = 980,
    [double]$PreRingTau = 36,
    [int]$ExitMs = 4000,
    [switch]$ProbeFinalizeAware,
    [int]$DitherBitDepth = 24,
    [int]$PostLoadDitherBitDepth = 16,
    [int]$PostLoadDelayMs = 250,
    [string]$NoiseShaper = "fixed4",
    [int]$IrReloadCount = 0,
    [int]$IrReloadIntervalMs = 300,
    [int]$BypassBurstCount = 0,
    [int]$BypassBurstIntervalMs = 40,
    [int]$BypassBurstValue = 0,
    [int]$IntentBurstCount = 0,
    [int]$IntentBurstIntervalMs = 25
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

if (-not (Test-Path -LiteralPath $SessionScriptPath)) {
    throw "Session script not found: $SessionScriptPath"
}

if ($BeginOnly -and $EndOnly) {
    throw "BeginOnly and EndOnly cannot be specified together."
}

if ($BeginOnly) {
    & $SessionScriptPath -Begin
    exit 0
}

if ($EndOnly) {
    & $SessionScriptPath -End
    exit 0
}

if (-not (Test-Path -LiteralPath $ExePath)) {
    throw "Executable not found: $ExePath"
}

$existingProcess = Get-Process -Name "ConvoPeq" -ErrorAction SilentlyContinue
if ($null -ne $existingProcess) {
    Write-Output "[ISR-8.1 CLI] Terminating existing ConvoPeq process"
    $existingProcess | Stop-Process -Force
}

if (-not (Test-Path -LiteralPath $IrPath)) {
    throw "IR file not found: $IrPath"
}

$defaultLogPath = "c:\VSC_Project\ConvoPeq\build\ConvoPeq_artefacts\Release\ConvoPeq.log"

Write-Output "[ISR-8.1 CLI] Begin snapshot"
& $SessionScriptPath -Begin

Write-Output "[ISR-8.1 CLI] Launch app with CLI automation"
$logLastWriteBefore = if (Test-Path -LiteralPath $defaultLogPath) { (Get-Item -LiteralPath $defaultLogPath).LastWriteTime } else { $null }
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

if ($ProbeFinalizeAware) {
    $appArgs += @(
        "--cli-noise-shaper", $NoiseShaper,
        "--cli-dither-bit-depth", ([string]$DitherBitDepth),
        "--cli-post-load-dither-bit-depth", ([string]$PostLoadDitherBitDepth),
        "--cli-post-load-delay-ms", ([string]$PostLoadDelayMs)
    )

    if ($IrReloadCount -le 0) {
        $IrReloadCount = 12
    }

    $appArgs += @(
        "--cli-ir-reload-count", ([string]$IrReloadCount),
        "--cli-ir-reload-interval-ms", ([string]$IrReloadIntervalMs)
    )

    if ($BypassBurstCount -le 0) {
        $BypassBurstCount = 40
    }

    $appArgs += @(
        "--cli-bypass-burst-count", ([string]$BypassBurstCount),
        "--cli-bypass-burst-interval-ms", ([string]$BypassBurstIntervalMs),
        "--cli-bypass-burst-value", ([string]$BypassBurstValue)
    )

    if ($IntentBurstCount -le 0) {
        $IntentBurstCount = 45
    }

    $appArgs += @(
        "--cli-intent-burst-count", ([string]$IntentBurstCount),
        "--cli-intent-burst-interval-ms", ([string]$IntentBurstIntervalMs)
    )

    Write-Output "[ISR-8.1 CLI] Finalize-aware probe enabled"
}

$process = Start-Process -FilePath $ExePath -ArgumentList $appArgs -PassThru
$timeoutSec = [Math]::Max(([int][Math]::Ceiling($ExitMs / 1000.0) + 20), 30)
[void]$process.WaitForExit($timeoutSec * 1000)
if (-not $process.HasExited) {
    throw "ConvoPeq did not exit within timeout (${timeoutSec}s)."
}

$appExitCode = $process.ExitCode
$logLastWriteAfter = if (Test-Path -LiteralPath $defaultLogPath) { (Get-Item -LiteralPath $defaultLogPath).LastWriteTime } else { $null }
Write-Output "[ISR-8.1 CLI] App exit code: $appExitCode"
Write-Output "[ISR-8.1 CLI] Log lastWrite before/after: $logLastWriteBefore -> $logLastWriteAfter"

if (-not (Test-Path -LiteralPath $defaultLogPath)) {
    Write-Output "[ISR-8.1 CLI] Waiting for log file to appear: $defaultLogPath"
    for ($i = 0; $i -lt 2000; $i++) {
        if (Test-Path -LiteralPath $defaultLogPath) {
            break
        }
    }
}

Write-Output "[ISR-8.1 CLI] End delta"
& $SessionScriptPath -End
