@echo off
setlocal
set "DB=C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Tools\MSVC\14.51.36231\bin\Hostx64\x64\dumpbin.exe"
"%DB%" /SYMBOLS "C:\VSC_Project\ConvoPeq\build\ConvoPeq_artefacts\Release\ConvoPeq.exe" | findstr /i "21b8"
echo ---
echo Now finding with DISASM...
"%DB%" /DISASM "C:\VSC_Project\ConvoPeq\build\ConvoPeq_artefacts\Release\ConvoPeq.exe" | findstr /i "21b8d81"
