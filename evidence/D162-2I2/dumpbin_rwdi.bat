@echo off
setlocal EnableExtensions
set VCVARS="C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
call %VCVARS% >nul 2>&1
dumpbin /disasm "C:\VSC_Project\ConvoPeq\build-diag\CMakeFiles\AudioEngineHarness.dir\RelWithDebInfo\src\audioengine\ISRRuntimePublicationCoordinator.cpp.obj" > "C:\VSC_Project\ConvoPeq\evidence\D162-2I2\isrcoordinator_rwdi_disasm.txt" 2>&1
echo DONE
