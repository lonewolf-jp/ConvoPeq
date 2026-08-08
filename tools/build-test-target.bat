@echo off
setlocal EnableExtensions
REM ============================================================================
REM build-test-target.bat - テストターゲットのビルド・実行
REM   Usage: build-test-target.bat <TargetName>
REM ============================================================================
set "TARGET=%1"
if "%TARGET%"=="" set "TARGET=MpscBoundedRingTests"

call "%ProgramFiles%\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul 2>&1
call "%ProgramFiles(x86)%\Intel\oneAPI\setvars.bat" intel64 >nul 2>&1

echo === RECONFIGURE ===
cmake -S C:\VSC_Project\ConvoPeq -B C:\VSC_Project\ConvoPeq\build >nul 2>&1
echo === BUILD %TARGET% ===
cmake --build C:\VSC_Project\ConvoPeq\build --config Debug --target %TARGET% 2>&1 | findstr /i "error C[0-9] error LNK .exe"
if errorlevel 1 exit /b 1
exit /b 0
