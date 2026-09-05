@echo off
setlocal EnableExtensions
set VCVARS="C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
call %VCVARS% >nul 2>&1
dumpbin /headers "C:\VSC_Project\ConvoPeq\build-diag\CMakeFiles\AudioEngineHarness.dir\Release\src\audioengine\ISRRuntimePublicationCoordinator.cpp.obj" | findstr /C:"pointer to symbol" /C:"number of symbols" /C:"section header" /C:".text" > "C:\VSC_Project\ConvoPeq\evidence\D162-2I2\isrcoordinator_hdr.txt"
dumpbin /symbols "C:\VSC_Project\ConvoPeq\build-diag\CMakeFiles\AudioEngineHarness.dir\Release\src\audioengine\ISRRuntimePublicationCoordinator.cpp.obj" | findstr /C:"005" /C:"SECT" > "C:\VSC_Project\ConvoPeq\evidence\D162-2I2\isrcoordinator_syms.txt"
echo DONE
