@echo off
setlocal EnableExtensions EnableDelayedExpansion
set "CFG=%~1"
if "%CFG%"=="" set "CFG=Debug"

call "C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" x64
if errorlevel 1 echo GATB_VCVARS_FAIL rc=%errorlevel%
if errorlevel 1 exit /b 1

set IPP_INC="C:\Program Files (x86)\Intel\oneAPI\2026.1\include"
set "CL=/I %IPP_INC%"

cd /d C:\VSC_Project\ConvoPeq
if errorlevel 1 echo GATB_CWD_FAIL rc=%errorlevel%
if errorlevel 1 exit /b 1

echo GATB_BUILD_START config=%CFG% > "evidence\D135-8-9_GATE_B_%CFG%_BUILD_LOG.txt"
cmake --build build --config %CFG% >> "evidence\D135-8-9_GATE_B_%CFG%_BUILD_LOG.txt" 2>&1
echo GATB_CMAKE_EXIT=%errorlevel%
exit /b %errorlevel%
