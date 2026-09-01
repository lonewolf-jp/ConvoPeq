@echo off
setlocal EnableExtensions EnableDelayedExpansion
set "CFG=%~1"
if "%CFG%"=="" set "CFG=Debug"
set "LOG=%~2"
if "%LOG%"=="" set "LOG=evidence\T3C_%CFG%_BUILD_LOG.txt"

call "C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" x64
if errorlevel 1 echo T3CV_VCVARS_FAIL rc=%errorlevel%
if errorlevel 1 exit /b 1

set IPP_INC="C:\Program Files (x86)\Intel\oneAPI\2026.1\include"
set "CL=/I %IPP_INC%"

cd /d C:\VSC_Project\ConvoPeq
if errorlevel 1 echo T3CV_CWD_FAIL rc=%errorlevel%
if errorlevel 1 exit /b 1

echo T3C_BUILD_START config=%CFG% > "%LOG%"
cmake --build build --config %CFG% >> "%LOG%" 2>&1
echo T3C_CMAKE_EXIT=%errorlevel%
exit /b %errorlevel%
