@echo off
setlocal enabledelayedexpansion

call "%ProgramFiles%\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" x64
if %errorlevel% neq 0 exit /b 1

call "%ProgramFiles(x86)%\Intel\oneAPI\setvars.bat" intel64

echo === Step 1: Clean old build-icx artifacts ===
if exist "C:\VSC_Project\ConvoPeq\build-icx" (
    rmdir /s /q "C:\VSC_Project\ConvoPeq\build-icx"
    echo Cleaned build-icx
)

echo === Step 2: CMake Configure (icx + DIAG) ===
cmake -S C:\VSC_Project\ConvoPeq -B C:\VSC_Project\ConvoPeq\build-icx -G Ninja -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icx -DCONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS=ON
if %errorlevel% neq 0 (
    echo [FATAL] CMake configure failed
    exit /b 1
)

echo === Step 3: CMake Build (Release) ===
cmake --build C:\VSC_Project\ConvoPeq\build-icx --config Release 2>&1
set BUILD_EXIT=%errorlevel%

echo === Build exit code: %BUILD_EXIT% ===

echo.
echo === FILTERED WARNINGS (excluding nodiscard) ===
findstr /n "warning" C:\VSC_Project\ConvoPeq\build-icx\build-log.txt 2>nul | findstr /v "nodiscard" 2>nul

exit /b %BUILD_EXIT%
