@echo off
setlocal EnableExtensions EnableDelayedExpansion
REM Gate C-F1 build helper. Usage: gatef1_build.bat <SourceDir> <BuildDir> <Config> <LogPath>
REM Sources MSVC env (call) + oneAPI CL include, configures (no-op if cached) and builds.
set "SRCDIR=%~1"
set "BLDDIR=%~2"
set "CFG=%~3"
set "LOG=%~4"

call "C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" x64
if errorlevel 1 echo GATF1_VCVARS_FAIL rc=%errorlevel%
if errorlevel 1 exit /b 1

set IPP_INC="C:\Program Files (x86)\Intel\oneAPI\2026.1\include"
set "CL=/I %IPP_INC%"

echo GATF1_CONFIGURE_START src=%SRCDIR% bld=%BLDDIR% cfg=%CFG% > "%LOG%"
cmake -S "%SRCDIR%" -B "%BLDDIR%" -G "Ninja Multi-Config" -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl -DCMAKE_EXPORT_COMPILE_COMMANDS=ON >> "%LOG%" 2>&1
if errorlevel 1 echo GATF1_CONFIGURE_FAIL rc=%errorlevel%
if errorlevel 1 exit /b 1

echo GATF1_BUILD_START >> "%LOG%"
cmake --build "%BLDDIR%" --config %CFG% >> "%LOG%" 2>&1
echo GATF1_CMAKE_EXIT=%errorlevel%
exit /b %errorlevel%
