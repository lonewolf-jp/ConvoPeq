@echo off
setlocal enabledelayedexpansion
call "%ProgramFiles%\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" x64
if %errorlevel% neq 0 exit /b 1
call "%ProgramFiles(x86)%\Intel\oneAPI\setvars.bat" intel64

set "BUILD_DIR=C:\VSC_Project\ConvoPeq\build-icx"
set "JOBS=4"

:configure
if not exist "!BUILD_DIR!\CMakeCache.txt" (
    echo === CMake Configure (icx + DIAG) ===
    cmake -S C:\VSC_Project\ConvoPeq -B "!BUILD_DIR!" -G Ninja -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icx -DCONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS=ON
    if !errorlevel! neq 0 (
        echo [RETRY] Configure failed, clean stale cache and retry...
        if exist "!BUILD_DIR!" rmdir /s /q "!BUILD_DIR!"
        goto configure
    )
) else (
    echo === CMake cache found, skip configure ===
)

echo === Build icx Release (j!JOBS!, OOM fallback j1) ===
cd /d "!BUILD_DIR!"
ninja -j!JOBS!
if errorlevel 1 (
    echo [RETRY] OOM fallback: retrying with j1...
    ninja -j1
)
set "EX=!errorlevel!"
echo === ICX BUILD EXIT: !EX! ===
exit /b !EX!
