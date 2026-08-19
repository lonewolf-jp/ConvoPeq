@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64
if errorlevel 1 exit /b 1
echo VCVARSALL OK
cd /d C:\VSC_Project\ConvoPeq
cmake --build build --config Debug --target RetireGraceSemanticsTests 2>&1
if errorlevel 1 exit /b 1
echo BUILD OK
ctest -C Debug -R RetireGraceSemantics --output-on-failure 2>&1
