@echo off
chcp 65001 >nul 2>&1
call "C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
cd /d C:\VSC_Project\ConvoPeq\build
ctest -C Debug --output-on-failure
