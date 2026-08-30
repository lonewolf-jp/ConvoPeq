@echo off
setlocal EnableExtensions EnableDelayedExpansion
REM Gate C CTest helper. Usage: gatec_ctest.bat [Debug|Release]
REM Mirrors tools/run-ctest-check.bat env (vcvarsall x64 + oneAPI setvars, non-fatal)
REM and tools/run-ctest-full.bat invocation, but runs ALL registered tests (-E NOT applied;
REM the two known-excluded tests are recorded and classified in the Gate C report).
set "CFG=%~1"
if "%CFG%"=="" set "CFG=Debug"

call "C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" x64
if errorlevel 1 echo GATC_VCVARS_FAIL rc=%errorlevel%
if errorlevel 1 exit /b 1

call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" intel64
if errorlevel 1 echo GATC_ONEAPI_WARN rc=%errorlevel%

cd /d C:\VSC_Project\ConvoPeq\build
if errorlevel 1 echo GATC_CWD_FAIL rc=%errorlevel%
if errorlevel 1 exit /b 1

echo GATC_CTEST_START config=%CFG% > "C:\VSC_Project\ConvoPeq\evidence\D135-8-9_GATE_C_%CFG%_CTEST_LOG.txt"
ctest -C %CFG% --output-on-failure >> "C:\VSC_Project\ConvoPeq\evidence\D135-8-9_GATE_C_%CFG%_CTEST_LOG.txt" 2>&1
echo GATC_CTEST_EXIT=%errorlevel%
exit /b %errorlevel%
