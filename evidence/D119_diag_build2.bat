@echo off
setlocal EnableExtensions
set VCVARS="C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
set IPP_INC="C:\Program Files (x86)\Intel\oneAPI\2026.1\include"
call %VCVARS% >nul 2>&1
if errorlevel 1 (
    echo [ERROR] vcvars64 failed
    exit /b 1
)
set "CL=/I %IPP_INC%"
cd /d C:\VSC_Project\ConvoPeq
cmake -S . -B build-diag -G "Ninja Multi-Config" -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS=ON
if errorlevel 1 (
    echo [ERROR] CMake configure failed
    exit /b 1
)
cmake -S . -B build-diag -G "Ninja Multi-Config" -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS=ON
if errorlevel 1 (
    echo [ERROR] CMake reconfigure failed
    exit /b 1
)
findstr /C:"CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS:BOOL=ON" build-diag\CMakeCache.txt >nul
if errorlevel 1 (
    echo [ERROR] DIAGNOSTICS option not ON in cache
    exit /b 1
)
cmake --build build-diag --config Release
if errorlevel 1 (
    echo [ERROR] Build failed
    exit /b 1
)
echo [OK] DIAG build complete
exit /b 0
