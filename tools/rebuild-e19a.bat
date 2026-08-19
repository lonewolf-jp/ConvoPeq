@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64
if errorlevel 1 exit /b 1
echo VCVARSALL OK
cd /d C:\VSC_Project\ConvoPeq
echo Rebuilding changed files...
ninja -C build ConvoPeq 2>&1
if errorlevel 1 exit /b 1
echo BUILD OK
