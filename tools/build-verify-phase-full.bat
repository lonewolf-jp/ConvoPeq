@echo off
setlocal EnableExtensions
REM ============================================================================
REM build-verify-phase-full.bat - 実装検証用 Debug ビルド（全出力をログに記録）
REM   Usage: build-verify-phase-full.bat [target] [logfile]
REM ============================================================================
set "TARGET=%1"
if "%TARGET%"=="" set "TARGET=ConvoPeq"
set "LOGFILE=%2"
if "%LOGFILE%"=="" set "LOGFILE=C:\VSC_Project\ConvoPeq\build\verify-phase.log"

call "%ProgramFiles%\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul 2>&1
call "%ProgramFiles(x86)%\Intel\oneAPI\setvars.bat" intel64 >nul 2>&1

echo === TARGET: %TARGET% === > "%LOGFILE%"
cmake --build C:\VSC_Project\ConvoPeq\build --config Debug --target %TARGET% >> "%LOGFILE%" 2>&1
echo === EXIT: %ERRORLEVEL% === >> "%LOGFILE%"
exit /b %ERRORLEVEL%
