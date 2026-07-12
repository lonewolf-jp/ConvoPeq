@echo off
setlocal enabledelayedexpansion

call "%ProgramFiles%\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" x64
if %errorlevel% neq 0 (
    echo [ERROR] vcvarsall.bat failed
    exit /b 1
)

call "%ProgramFiles(x86)%\Intel\oneAPI\setvars.bat" intel64
if %errorlevel% neq 0 (
    echo [WARN] Intel oneAPI setvars failed (non-fatal)
)

echo === Running CTest ===
cd /d C:\VSC_Project\ConvoPeq\build
if %errorlevel% neq 0 (
    echo [ERROR] build directory not found
    exit /b 1
)

ctest -C Debug --output-on-failure -E "BuildInputSemanticContract|RuntimeWorldAuthority"
set CTEST_EXIT=%errorlevel%

echo === CTest exit code: %CTEST_EXIT% ===
exit /b %CTEST_EXIT%
