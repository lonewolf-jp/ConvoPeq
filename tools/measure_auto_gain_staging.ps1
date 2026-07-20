<#
.SYNOPSIS
    Auto Gain Staging 自動測定スクリプト v14.47
.DESCRIPTION
    全IRに対して自動ゲインステージングのPlanDiagnosticsを収集し、
    クランプ発生率・boundExcessDb分布・Plan出力を集計します。
    結果は doc/work77/benchmark-results/ に出力されます。

    前提条件:
      - Release ビルドが存在すること (build\ConvoPeq_artefacts\Release\ConvoPeq.exe)
      - オーディオデバイスが利用可能であること
      - IRファイルが sampledata/ に配置済みであること
#>

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

# ─── 設定 ────────────────────────────────────────────────────────────────
$Exe = "C:\VSC_Project\ConvoPeq\build\ConvoPeq_artefacts\Release\ConvoPeq.exe"
$SyntheticDir = "C:\VSC_Project\ConvoPeq\sampledata\synthetic"
$RealDir      = "C:\VSC_Project\ConvoPeq\sampledata\real_iris\wavs"
$OutputDir    = "C:\VSC_Project\ConvoPeq\doc\work77\benchmark-results"
$RunDurationMs = 5000
$MaxConcurrent = 1   # 同時実行数（1推奨）

# ─── 結果格納用 ──────────────────────────────────────────────────────────
$Results = @()

# ─── 準備 ─────────────────────────────────────────────────────────────────
if (-not (Test-Path $Exe)) {
    Write-Error "Release exe not found: $Exe"
    exit 1
}

if (-not (Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
}

# ─── 単一IR測定 ──────────────────────────────────────────────────────────
function Measure-Ir {
    param([string]$IrPath, [string]$Label)

    $logFile = Join-Path $OutputDir "measure_$(Get-Random).log"
    $summaryFile = Join-Path $OutputDir "summary_$(Get-Random).json"

    Write-Host "  [MEASURE] $Label ... " -NoNewline

    # 過去プロセスを強制終了
    Get-Process -Name ConvoPeq* -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
    Start-Sleep -Milliseconds 500

    # 測定実行（PowerShell 5.1互換: 単一文字列で引数渡し）
    $argsStr = "--cli-run --cli-ir `"$IrPath`" --cli-rebuild --cli-exit-ms $RunDurationMs --cli-log-file `"$logFile`""
    $proc = Start-Process -FilePath $Exe -ArgumentList $argsStr -NoNewWindow -Wait -PassThru

    Start-Sleep -Seconds 2  # ログファイル書き込み完了待ち

    if (-not (Test-Path $logFile)) {
        Write-Host "NO LOG (exit=$($proc.ExitCode))" -ForegroundColor Red
        return @{ Label = $Label; ExitCode = $proc.ExitCode; Status = 'NO_LOG' }
    }

    # ログ解析
    $logContent = Get-Content $logFile -Raw

    # AUTO_GAIN_CLAMP (存在すれば clamp 発生)
    $clampEntries = [regex]::Matches($logContent, '\[AUTO_GAIN_CLAMP\].*').Value

    # boundExcessDb
    $boundMatches = [regex]::Matches($logContent, 'boundExcessDb=([-\d.]+)')
    $boundValues = @($boundMatches | ForEach-Object { [double]$_.Groups[1].Value })

    # CLI_PERF_RAW
    $perfLines = [regex]::Matches($logContent, '\[CLI_PERF_RAW\] callbacks=(\d+).*procTimeUsAvg=([\d.]+).*procTimeUsMax=([\d.]+).*blockSamples=(\d+)')
    $perfData = @($perfLines | ForEach-Object {
        @{
            Callbacks    = [int]$_.Groups[1].Value
            AvgUs        = [double]$_.Groups[2].Value
            MaxUs        = [double]$_.Groups[3].Value
            BlockSamples = [int]$_.Groups[4].Value
        }
    })

    # REBUILD_TELEMETRY (リビルド回数)
    $rebuildCount = [regex]::Matches($logContent, 'event=REBUILD_DISPATCHED').Count

    # PlanDiagnostics (AutoGainPlanner の出力)
    $planGainMatch = [regex]::Match($logContent, 'plan.*inputHeadroomDb=([-\d.]+).*outputMakeupDb=([-\d.]+)')
    $planOutput = if ($planGainMatch.Success) {
        @{ InputHeadroomDb = [double]$planGainMatch.Groups[1].Value; OutputMakeupDb = [double]$planGainMatch.Groups[2].Value }
    } else { $null }

    # IR情報
    $irLoadedMatch = [regex]::Match($logContent, 'isIRLoaded=(\d+) irLen=(\d+)')
    $irLoaded = if ($irLoadedMatch.Success) { [int]$irLoadedMatch.Groups[1].Value -eq 1 } else { $false }
    $irLen = if ($irLoadedMatch.Success) { [int]$irLoadedMatch.Groups[2].Value } else { 0 }

    # 結果まとめ
    $result = @{
        Label          = $Label
        ExitCode       = $proc.ExitCode
        Status         = if ($irLoaded) { 'OK' } else { 'NO_IR' }
        IrLoaded       = $irLoaded
        IrLen          = $irLen
        ClampCount     = $clampEntries.Count
        BoundValues    = $boundValues
        AvgProcTimeUs  = if ($perfData.Count -gt 0) { ($perfData | Measure-Object -Property AvgUs -Average).Average } else { $null }
        MaxProcTimeUs  = if ($perfData.Count -gt 0) { ($perfData | Measure-Object -Property MaxUs -Maximum).Maximum } else { $null }
        TotalCallbacks = if ($perfData.Count -gt 0) { ($perfData | Measure-Object -Property Callbacks -Sum).Sum } else { 0 }
        RebuildCount   = $rebuildCount
        PlanOutput     = $planOutput
        LogFile        = $logFile
    }

    # サマリ表示
    if ($irLoaded) {
        Write-Host "IR=$irLen callbacks=$($result.TotalCallbacks) avgProc=$($result.AvgProcTimeUs)us clamps=$($result.ClampCount) rebuilds=$($result.RebuildCount)" -ForegroundColor Green
    } else {
        Write-Host "IR FAILED" -ForegroundColor Yellow
    }

    # 結果をJSON保存
    $result | ConvertTo-Json -Compress | Set-Content $summaryFile -Encoding UTF8

    return $result
}

# ─── IR一覧収集 ─────────────────────────────────────────────────────────
$allIrs = @()

# シンセティックIR (直下の .wav)
Get-ChildItem -Path $SyntheticDir -Filter '*.wav' | ForEach-Object {
    $allIrs += @{ Path = $_.FullName; Label = "synth/$($_.BaseName)" }
}
# シンセティックIR (サブフォルダ)
Get-ChildItem -Path $SyntheticDir -Recurse -Filter '*.wav' | Where-Object { $_.DirectoryName -ne $SyntheticDir } | ForEach-Object {
    $relDir = $_.Directory.Name
    $allIrs += @{ Path = $_.FullName; Label = "synth/$relDir/$($_.BaseName)" }
}
# 実IR (全て)
if (Test-Path $RealDir) {
    Get-ChildItem -Path $RealDir -Recurse -Filter '*.wav' | ForEach-Object {
        $allIrs += @{ Path = $_.FullName; Label = "real/$($_.BaseName)" }
    }
}

Write-Host "=== Auto Gain Staging 自動測定 ===" -ForegroundColor Cyan
Write-Host "IR数: $($allIrs.Count)"
Write-Host "出力先: $OutputDir"
Write-Host "実行時間/IR: ${RunDurationMs}ms"
Write-Host ""

# ─── 測定実行 (逐次) ─────────────────────────────────────────────────────
$totalStart = Get-Date
$successCount = 0
$failCount = 0

for ($i = 0; $i -lt $allIrs.Count; $i++) {
    $ir = $allIrs[$i]
    $progress = "[$($i+1)/$($allIrs.Count)]"
    Write-Host "$progress $($ir.Label)" -ForegroundColor White

    $result = Measure-Ir -IrPath $ir.Path -Label $ir.Label
    $Results += $result

    if ($result.Status -eq 'OK') {
        $successCount++
    } else {
        $failCount++
    }

    # IR間のクールダウン
    Start-Sleep -Milliseconds 500
}

$totalDuration = (Get-Date) - $totalStart

# ─── 集計レポート ─────────────────────────────────────────────────────────
Write-Host ""
Write-Host "=== 集計レポート ===" -ForegroundColor Cyan
Write-Host "成功: $successCount / $($allIrs.Count)"
Write-Host "失敗: $failCount"
Write-Host "所要時間: $($totalDuration.TotalMinutes)F2 分"

# クランプ統計
$clampResults = $Results | Where-Object { $_.ClampCount -gt 0 }
Write-Host ""
Write-Host "--- クランプ発生IR ($($clampResults.Count)件) ---" -ForegroundColor Yellow
$clampResults | ForEach-Object {
    Write-Host "  $($_.Label): $($_.ClampCount) clamps"
}

# boundExcessDb 統計
$allBounds = $Results | ForEach-Object { $_.BoundValues } | Where-Object { $_ -ne $null }
if ($allBounds.Count -gt 0) {
    $boundStats = $allBounds | Measure-Object -Minimum -Maximum -Average
    Write-Host ""
    Write-Host "--- boundExcessDb 分布 ---" -ForegroundColor Yellow
    Write-Host "  最小: $($boundStats.Minimum)F2 dB"
    Write-Host "  最大: $($boundStats.Maximum)F2 dB"
    Write-Host "  平均: $($boundStats.Average)F2 dB"
    Write-Host "  サンプル数: $($allBounds.Count)"

    # パーセンタイル
    $sorted = $allBounds | Sort-Object
    Write-Host "  中央値: $($sorted[$sorted.Count/2])F2 dB"
    Write-Host "  P90: $($sorted[$sorted.Count*0.9])F2 dB"
    Write-Host "  P95: $($sorted[$sorted.Count*0.95])F2 dB"
    Write-Host "  P99: $($sorted[$sorted.Count*0.99])F2 dB"
}

# Plan出力
$planResults = $Results | Where-Object { $_.PlanOutput -ne $null }
if ($planResults.Count -gt 0) {
    Write-Host ""
    Write-Host "--- PlanDiagnostics サンプル ---" -ForegroundColor Yellow
    $planResults | Select-Object -First 5 | ForEach-Object {
        Write-Host "  $($_.Label): inputHeadroom=$($_.PlanOutput.InputHeadroomDb)dB outputMakeup=$($_.PlanOutput.OutputMakeupDb)dB"
    }
}

# 処理時間統計
$procTimes = $Results | Where-Object { $_.AvgProcTimeUs -ne $null } | Select-Object -ExpandProperty AvgProcTimeUs
if ($procTimes.Count -gt 0) {
    $procStats = $procTimes | Measure-Object -Minimum -Maximum -Average
    Write-Host ""
    Write-Host "--- 処理時間 (平均, μs) ---" -ForegroundColor Yellow
    Write-Host "  最小: $($procStats.Minimum)F2"
    Write-Host "  最大: $($procStats.Maximum)F2"
    Write-Host "  平均: $($procStats.Average)F2"
}

# ─── 結果保存 ─────────────────────────────────────────────────────────────
$reportFile = Join-Path $OutputDir "auto_gain_staging_report_$(Get-Date -Format 'yyyyMMdd_HHmmss').json"
$report = @{
    Timestamp       = (Get-Date -Format 'o')
    TotalIRs        = $allIrs.Count
    SuccessCount    = $successCount
    FailCount       = $failCount
    DurationSeconds = $totalDuration.TotalSeconds
    Results         = $Results
}
$report | ConvertTo-Json -Depth 10 | Set-Content $reportFile -Encoding UTF8
Write-Host ""
Write-Host "レポート保存: $reportFile" -ForegroundColor Green
Write-Host "=== 測定完了 ===" -ForegroundColor Cyan
