@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64
if errorlevel 1 exit /b 1
cd /d C:\VSC_Project\ConvoPeq\build
ctest -C Debug --output-on-failure -E "BuildInputSemanticContract|RuntimeWorldAuthority" 2>&1
