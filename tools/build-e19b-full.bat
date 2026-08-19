@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64
if errorlevel 1 exit /b 1
cd /d C:\VSC_Project\ConvoPeq
cmake --build build --config Debug 2>&1
if errorlevel 1 exit /b 1
echo FULL BUILD OK
