@echo off
setlocal EnableExtensions EnableDelayedExpansion
set VCVARS="C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
set IPP_INC="C:\Program Files (x86)\Intel\oneAPI\2026.1\include"
call %VCVARS% >nul 2>&1
if errorlevel 1 (
    echo [ERROR] vcvars64 failed
    exit /b 1
)
set "CL=/I %IPP_INC%"
echo [INFO] vcvars + ipp.h include set
cd /d C:\VSC_Project\ConvoPeq
build.bat %*
