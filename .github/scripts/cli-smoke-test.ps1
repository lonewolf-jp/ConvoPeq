param(
    [string]$ExePath = "c:\VSC_Project\ConvoPeq\build\ConvoPeq_artefacts\Debug\ConvoPeq.exe",
    [int]$ExitMs = 1500,
    [int]$TimeoutSec = 30,
    [switch]$UseReleaseLog,
    [switch]$KillExisting,
    [switch]$RequireAudioCallbacks
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

if (-not (Test-Path -LiteralPath $ExePath)) {
    throw "Executable not found: $ExePath"
}

$logPath = if ($UseReleaseLog) {
    "c:\VSC_Project\ConvoPeq\build\ConvoPeq_artefacts\Release\ConvoPeq.log"
}
else {
    "c:\VSC_Project\ConvoPeq\build\ConvoPeq_artefacts\Debug\ConvoPeq.log"
}

if ($KillExisting) {
    $existing = Get-Process -Name "ConvoPeq" -ErrorAction SilentlyContinue
    if ($null -ne $existing) {
        Write-Output "[CLI-SMOKE] Terminating existing ConvoPeq process"
        $existing | Stop-Process -Force
    }

    if (Test-Path -LiteralPath $logPath) {
        Remove-Item -LiteralPath $logPath -Force
    }
}

$beforeLineCount = 0
if (Test-Path -LiteralPath $logPath) {
    $beforeLineCount = [int](@(Get-Content -LiteralPath $logPath).Count)
}

Write-Output "[CLI-SMOKE] Launch: $ExePath"
$appArgs = @(
    "--cli-run",
    "--cli-learning-action", "start",
    "--cli-learning-mode", "short",
    "--cli-exit-ms", ([string]$ExitMs)
)

$proc = Start-Process -FilePath $ExePath -ArgumentList $appArgs -PassThru
[void]$proc.WaitForExit($TimeoutSec * 1000)
if (-not $proc.HasExited) {
    try { $proc.Kill() } catch {}
    throw "ConvoPeq did not exit within timeout (${TimeoutSec}s)."
}

$exitCode = $proc.ExitCode
Write-Output "[CLI-SMOKE] App exit code: $exitCode"
if ($exitCode -ne 0) {
    throw "CLI smoke failed: app exit code=$exitCode"
}

if (-not (Test-Path -LiteralPath $logPath)) {
    throw "CLI smoke failed: log file not found: $logPath"
}

$automationRequested = 0
$autoExitScheduled = 0
$audioCallbacks = 0
$learningCommandQueued = 0

# logger flush のタイミング差を吸収するため、短時間ポーリングする。
for ($attempt = 0; $attempt -lt 20; $attempt++) {
    $allLines = @(Get-Content -LiteralPath $logPath)

    # 起動時にログがローテーション/再生成される場合があるため、
    # after < before のときは全行を検査対象にする。
    $linesToInspect = @(
        if ($allLines.Count -lt $beforeLineCount) {
            $allLines
        }
        elseif ($beforeLineCount -lt $allLines.Count) {
            $allLines[$beforeLineCount..($allLines.Count - 1)]
        }
        else {
            @()
        }
    )

    # 行数が変わらないまま内容更新されるケースに備えて、
    # 窓が空なら末尾400行をフォールバックとして検査する。
    if ($linesToInspect.Count -eq 0 -and $allLines.Count -gt 0) {
        $start = [Math]::Max(0, $allLines.Count - 400)
        $linesToInspect = @($allLines[$start..($allLines.Count - 1)])
    }

    $automationRequested = @($linesToInspect | Where-Object { $_ -match '\[CLI\]\s+Automation requested:' }).Count
    $autoExitScheduled = @($linesToInspect | Where-Object { $_ -match '\[CLI\]\s+Auto-exit scheduled in\s+' }).Count
    $audioCallbacks = @($linesToInspect | Where-Object { $_ -match '\[CLI_PERF_RAW\]\s+callbacks=([1-9][0-9]*)\b' }).Count
    $learningCommandQueued = @($linesToInspect | Where-Object { $_ -match '\[AudioEngine\]\s+startNoiseShaperLearning:\s+command queued\s+mode=' }).Count

    if ($automationRequested -ge 1 -and $autoExitScheduled -ge 1 -and $learningCommandQueued -ge 1 -and (-not $RequireAudioCallbacks -or $audioCallbacks -ge 1)) {
        break
    }

    Start-Sleep -Milliseconds 200
}

if ($automationRequested -lt 1) {
    throw "CLI smoke failed: '[CLI] Automation requested:' was not found in inspected log window."
}

if ($autoExitScheduled -lt 1) {
    throw "CLI smoke failed: '[CLI] Auto-exit scheduled in' was not found in inspected log window."
}

if ($learningCommandQueued -lt 1) {
    throw "CLI smoke failed: '[AudioEngine] startNoiseShaperLearning: command queued ...' was not found in inspected log window."
}

if ($RequireAudioCallbacks -and $audioCallbacks -lt 1) {
    throw "CLI smoke failed: '[CLI_PERF_RAW] callbacks=' with a positive callback count was not found in inspected log window."
}

Write-Output "[CLI-SMOKE] PASS"
Write-Output "[CLI-SMOKE] logPath=$logPath"
Write-Output "[CLI-SMOKE] automationRequested=$automationRequested"
Write-Output "[CLI-SMOKE] autoExitScheduled=$autoExitScheduled"
Write-Output "[CLI-SMOKE] learningCommandQueued=$learningCommandQueued"
if ($RequireAudioCallbacks) {
    Write-Output "[CLI-SMOKE] audioCallbacks=$audioCallbacks"
}
