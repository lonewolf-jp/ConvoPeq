@echo off
REM ==========================================================================
REM Auto Gain Staging Benchmark Runner
REM ==========================================================================
REM 使用方法:
REM   run_benchmark.bat              # デフォルト設定ですべての IR をテスト
REM   run_benchmark.bat --quick      # クイックテスト（3 IR のみ）
REM   run_benchmark.bat --single <file>  # 単一 IR でテスト
REM
REM 結果は doc/work77/benchmark-results/ に出力されます
REM ==========================================================================

setlocal enabledelayedexpansion

set EXE=C:\VSC_Project\ConvoPeq\build\ConvoPeq_artefacts\Release\ConvoPeq.exe
set SYNTH_DIR=C:\VSC_Project\ConvoPeq\sampledata\synthetic
set REAL_DIR=C:\VSC_Project\ConvoPeq\sampledata\real_iris\wavs
set OUTPUT_DIR=C:\VSC_Project\ConvoPeq\doc\work77\benchmark-results

if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

echo ===========================================================================
echo Auto Gain Staging Benchmark Runner
echo ===========================================================================
echo.
echo Exe: %EXE%
echo Synthetic IRs: %SYNTH_DIR%
echo Real IRs: %REAL_DIR%
echo Output: %OUTPUT_DIR%
echo.

REM ─── Run benchmark for each IR ────────────────────────────────────────────
REM Each run: load IR, process for 5 seconds, then exit

set MODE=%1

if "%MODE%"=="--quick" goto :quick
if "%MODE%"=="--single" goto :single
goto :all

:quick
echo [QUICK MODE] Testing synthetic IRs only...
for %%f in ("%SYNTH_DIR%\dirac_k2.wav" "%SYNTH_DIR%\minimum_phase\minphase_bandpass_2k_12dB.wav" "%SYNTH_DIR%\linear_phase\linphase_bandpass_2k_12dB.wav") do (
    echo Loading: %%~nxf
    start /wait "" "%EXE%" --cli-run --cli-ir "%%f" --cli-exit-ms 5000
    echo   Exit code: !ERRORLEVEL!
    echo.
)
goto :end

:single
set SINGLE_IR=%2
if "%SINGLE_IR%"=="" (
    echo Usage: run_benchmark.bat --single ^<file^>
    exit /b 1
)
echo [SINGLE MODE] Testing: %SINGLE_IR%
start /wait "" "%EXE%" --cli-run --cli-ir "%SINGLE_IR%" --cli-exit-ms 5000
echo   Exit code: !ERRORLEVEL!
goto :end

:all
echo [FULL MODE] Testing all IRs (this will take a while)...
echo.
echo Step 1: Synthetic IRs...
for %%f in ("%SYNTH_DIR%\*.wav") do (
    echo   %%f
    start /wait "" "%EXE%" --cli-run --cli-ir "%%f" --cli-exit-ms 3000
)
for /d %%d in ("%SYNTH_DIR%\*") do (
    for %%f in ("%%d\*.wav") do (
        echo   %%f
        start /wait "" "%EXE%" --cli-run --cli-ir "%%f" --cli-exit-ms 3000
    )
)

echo.
echo Step 2: Real IRs (sampling every 5th IR for speed)...
set COUNT=0
for /r "%REAL_DIR%" %%f in (*.wav) do (
    set /a COUNT+=1
    set /a MOD=!COUNT! %% 5
    if !MOD!==0 (
        echo   %%f
        start /wait "" "%EXE%" --cli-run --cli-ir "%%f" --cli-exit-ms 3000
    )
)

:end
echo.
echo ===========================================================================
echo Benchmark complete.
echo Check OutputDebugString / DebugView for diagnostic results.
echo Synthetic IR results: %SYNTH_DIR%
echo Real IR results: %REAL_DIR%
echo ===========================================================================
endlocal
