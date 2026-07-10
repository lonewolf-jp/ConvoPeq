@echo off
REM ============================================================
REM ConvoPeq WPR トレース測定スクリプト
REM 管理者として実行してください
REM ============================================================
setlocal
set WPR="C:\Program Files (x86)\Windows Kits\10\Windows Performance Toolkit\wpr.exe"
set WPA="C:\Program Files (x86)\Windows Kits\10\Windows Performance Toolkit\wpa.exe"
set PROFILE=%~dp0..\tools\convopeq-trace.wprp
set OUTPUT=%~dp0ConvoPeqTrace.etl

echo =============================================
echo  ConvoPeq WPR トレース測定
echo =============================================
echo.
echo 手順1: トレース開始
echo   %WPR% -start %PROFILE%
echo.
echo 手順2: ConvoPeq を起動し操作
echo    (IR読込 / PEQ設定 / ANS / 音楽再生)
echo.
echo 手順3: 30秒〜60秒後にトレース停止
echo   %WPR% -stop %OUTPUT%
echo.
echo 手順4: WPA で解析
echo   %WPA% %OUTPUT%
echo.
echo === 以下のコマンドを1つずつ実行してください ===
echo.
cmd
