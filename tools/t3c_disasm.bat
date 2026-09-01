@echo off
setlocal
call "C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul 2>&1
if errorlevel 1 (echo DISASM_VCVARS_FAIL & exit /b 1)
cd /d C:\VSC_Project\ConvoPeq
dumpbin /disasm /NOLOGO build\t3c_lockfree_probe.exe > evidence\T3C_PROBE_DISASM.txt 2>&1
if errorlevel 1 (echo DISASM_FAIL & exit /b 1)
echo DISASM_OK
