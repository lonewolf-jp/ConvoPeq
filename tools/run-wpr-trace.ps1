# ConvoPeq WPR トレース測定スクリプト
# 管理者 PowerShell で実行してください

$WprExe = "C:\Program Files (x86)\Windows Kits\10\Windows Performance Toolkit\wpr.exe"
$WpaExe = "C:\Program Files (x86)\Windows Kits\10\Windows Performance Toolkit\wpa.exe"
$WprConfig = ".\tools\convopeq-trace.wprp"
$DataFile = ".\doc\work63\ConvoPeqTrace.etl"

Write-Host "======================================" -ForegroundColor Cyan
Write-Host " ConvoPeq WPR トレース測定" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan
Write-Host ""

# 管理者チェック
$isAdmin = [Security.Principal.WindowsPrincipal]::new([Security.Principal.WindowsIdentity]::GetCurrent())
if (-not $isAdmin.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Host "❌ 管理者権限が必要です。右クリック→「PowerShell を管理者として実行」" -ForegroundColor Red
    exit 1
}

# Step 1: 古いトレースをクリアして開始
Write-Host "[1/4] トレース開始..." -ForegroundColor Green
Start-Process $WprExe -ArgumentList "-cancel" -NoNewWindow -Wait | Out-Null
Start-Process $WprExe -ArgumentList "-start $WprConfig" -NoNewWindow -Wait

Write-Host "  ✅ トレース開始済み" -ForegroundColor Green
Write-Host ""
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Yellow
Write-Host " [2/4] ConvoPeq を起動して操作してください" -ForegroundColor Yellow
Write-Host "" -ForegroundColor Yellow
Write-Host "  起動: .\build\ConvoPeq_artefacts\Release\ConvoPeq.exe" -ForegroundColor Yellow
Write-Host "  操作:" -ForegroundColor Yellow
Write-Host "  1. IRファイル読み込み" -ForegroundColor Yellow
Write-Host "  2. PEQ設定" -ForegroundColor Yellow
Write-Host "  3. ANS Continuousモード開始" -ForegroundColor Yellow
Write-Host "  4. 音楽再生（30秒〜60秒程度）" -ForegroundColor Yellow
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Yellow
Write-Host ""
Read-Host "操作が終わったら Enter を押してください"

# Step 2: トレース停止
Write-Host "[3/4] トレース停止中..." -ForegroundColor Green
Start-Process $WprExe -ArgumentList "-stop $DataFile" -NoNewWindow -Wait
Write-Host "  ✅ 保存完了: $DataFile" -ForegroundColor Green

if (Test-Path $DataFile) {
    $size = (Get-Item $DataFile).Length / 1MB
    Write-Host "  📄 ファイルサイズ: $([math]::Round($size, 1)) MB" -ForegroundColor Cyan
}

Write-Host ""
Write-Host "[4/4] WPA で解析" -ForegroundColor Green
$ans = Read-Host "WPA を起動しますか？ (y/n)"
if ($ans -eq "y") {
    Start-Process $WpaExe -ArgumentList "`"$DataFile`""
    Write-Host "  ✅ WPA 起動" -ForegroundColor Green
}

Write-Host ""
Write-Host "===== 完了 =====" -ForegroundColor Cyan
Write-Host "ETL: $DataFile" -ForegroundColor Gray
