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
cmake --build build-diag --config Release --target AudioEngineHarness
if errorlevel 1 (
    echo [ERROR] Build failed
    exit /b 1
)
echo [OK] D162-2B baseline harness ConvoPeq build complete
exit /b 0
