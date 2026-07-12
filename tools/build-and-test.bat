@echo off
setlocal enabledelayedexpansion

echo === Step 1: Initialize VS + Intel environment ===
call "%ProgramFiles%\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" x64
if %errorlevel% neq 0 (
    echo [FATAL] vcvarsall.bat failed
    exit /b 1
)

call "%ProgramFiles(x86)%\Intel\oneAPI\setvars.bat" intel64
echo.

echo === Step 2: CMake Configure ===
cmake -S C:\VSC_Project\ConvoPeq -B C:\VSC_Project\ConvoPeq\build -G "Ninja Multi-Config" -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl
if %errorlevel% neq 0 (
    echo [FATAL] CMake configure failed
    exit /b 1
)
echo.

echo === Step 3: CMake Build (Debug) ===
cmake --build C:\VSC_Project\ConvoPeq\build --config Debug
if %errorlevel% neq 0 (
    echo [FATAL] Build failed
    exit /b 1
)
echo.

echo === Step 4: CTest ===
cd /d C:\VSC_Project\ConvoPeq\build
ctest -C Debug --output-on-failure -E "BuildInputSemanticContract|RuntimeWorldAuthority"
set CTEST_EXIT=%errorlevel%
echo.

echo === CTest finished with exit code: %CTEST_EXIT% ===
exit /b %CTEST_EXIT%
