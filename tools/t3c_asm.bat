@echo off
setlocal
call "C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul 2>&1
if errorlevel 1 (echo ASM_VCVARS_FAIL & exit /b 1)
cd /d C:\VSC_Project\ConvoPeq
cl /nologo /EHsc /std:c++17 /O2 /FAcs /Fabuild\t3c_probe.asm /Fobuild\ tools\t3c_lockfree_probe.cpp > build\t3c_asm_compile.txt 2>&1
if errorlevel 1 (echo ASM_COMPILE_FAIL & type build\t3c_asm_compile.txt & exit /b 1)
echo ASM_OK
