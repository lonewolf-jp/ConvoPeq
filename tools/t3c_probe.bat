@echo off
setlocal
call "C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul 2>&1
if errorlevel 1 (echo PROBE_VCVARS_FAIL & exit /b 1)
cd /d C:\VSC_Project\ConvoPeq
cl /nologo /EHsc /std:c++17 /O2 /Fe:build\t3c_lockfree_probe.exe /Fobuild\ tools\t3c_lockfree_probe.cpp > evidence\T3C_LOCKFREE_PROBE.txt 2>&1
if errorlevel 1 (echo PROBE_COMPILE_FAIL >> evidence\T3C_LOCKFREE_PROBE.txt & type evidence\T3C_LOCKFREE_PROBE.txt & exit /b 1)
build\t3c_lockfree_probe.exe >> evidence\T3C_LOCKFREE_PROBE.txt 2>&1
echo PROBE_EXIT=%errorlevel% >> evidence\T3C_LOCKFREE_PROBE.txt
type evidence\T3C_LOCKFREE_PROBE.txt
exit /b %errorlevel%
