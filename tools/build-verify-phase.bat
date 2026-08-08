@echo off
setlocal EnableExtensions
REM ============================================================================
REM build-verify-phase.bat - Phase 実装検証用 Debug ビルド（エラー・主要警告のみ表示）
REM   Usage: build-verify-phase.bat [target]
REM ============================================================================
set "TARGET=%1"
if "%TARGET%"=="" set "TARGET=ConvoPeq"

call "%ProgramFiles%\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul 2>&1
call "%ProgramFiles(x86)%\Intel\oneAPI\setvars.bat" intel64 >nul 2>&1

cmake --build C:\VSC_Project\ConvoPeq\build --config Debug --target %TARGET% 2>&1 | findstr /i "error C[0-9] error LNK warning C[0-9] .vcxproj ->"
exit /b %ERRORLEVEL%
