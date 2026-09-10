@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
set "CL=/I"C:\Program Files (x86)\Intel\oneAPI\2026.1\include""
cd /d C:\VSC_Project\ConvoPeq\build
ninja ConvoPeq:Debug -j2 || exit /b 1
ninja -j2 || exit /b 1
