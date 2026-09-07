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
if "%1"=="diag" goto DIAG
cmake --build build-diag --config Debug
if errorlevel 1 ( echo [ERROR] Debug build failed & exit /b 1 )
echo [OK] build-diag Debug complete
cmake --build build-diag --config Release
if errorlevel 1 ( echo [ERROR] Release build failed & exit /b 1 )
echo [OK] build-diag Release complete
goto END
:DIAG
cmake --build build-diag --config RelWithDebInfo
if errorlevel 1 ( echo [ERROR] RWDI build failed & exit /b 1 )
echo [OK] build-diag RelWithDebInfo complete
:END
exit /b 0
